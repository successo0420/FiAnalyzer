import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import streamlit as st

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide")  # Make the Streamlit layout full-width

# Try to load the SpaCy NLP model (used for merchant name extraction)
# If it's not installed, fallback to basic string handling.
try:
    nlp = spacy.load("en_core_web_sm")
except (OSError, ImportError):
    st.warning(
        "SpaCy model 'en_core_web_sm' not found. Merchant extraction will be basic. "
        "To install, run: python -m spacy download en_core_web_sm"
    )
    nlp = None

# Predefined budget categories
BUDGET_CATEGORIES = [
    "Merchandise", "Gasoline", "Payments and Credits", "Restaurants", "Supermarkets",
    "Services", "Department Stores", "Education", "Travel/ Entertainment",
    "Awards and Rebate Credits", "Automotive", "Medical Services", "Government Services",
]

# Default budget targets for each category (all 0 for now, can be filled later)
BUDGET_TARGETS = {cat: 0 for cat in BUDGET_CATEGORIES}

# Categories that shouldn’t be treated as negative expenses
SKIP_FOR_NEGATIVE = ["Payments and Credits"]
AWARDS_CATEGORY = "Awards and Rebate Credits"

# Aliases for common column names across different banks
# This helps auto-detect column names no matter how they appear in the CSV
COLUMN_ALIASES = {
    'date': ['trans. date', 'transaction date', 'date', 'posting date'],
    'description': ['description', 'details', 'merchant', 'payee'],
    'category': ['category', 'type', 'classification'],
    'amount': ['amount', 'value', 'price', 'debit/credit', 'debit/credit (amount)'],
    'deposit': ['deposit', 'deposits', 'credit', 'inflow'],
    'withdrawal': ['withdrawal', 'withdrawals', 'debit', 'outflow']
}

# Hardcoded last 4 digits of savings account (for transfer detection)
SAVINGS_NUMBER = "1234"


# ---------- HELPERS: COLUMN MAPPING ----------
def find_and_map_columns(df):
    """
    Auto-detects which columns in the uploaded CSV map to our standard names
    (date, description, category, amount, deposit, withdrawal).
    Returns a dict mapping standard_name -> actual column in df.
    """
    mapped_cols = {}
    # Dictionary: lowercase column name -> original column name
    df_cols_lower = {col.lower(): col for col in df.columns}

    for standard_name, aliases in COLUMN_ALIASES.items():
        mapped_cols[standard_name] = None  # Default is None (missing)
        for alias in aliases:
            if alias in df_cols_lower:
                mapped_cols[standard_name] = df_cols_lower[alias]  # Map alias to real col
                break
    return mapped_cols


# ---------- CLASSIFICATION ----------
def classify_transactions(df):
    """
    Classifies each transaction into Income, Expense, Transfer, etc.
    Also normalizes some categories based on rules.
    """
    df = df.copy()

    # Make sure description and category are strings (avoid dtype errors)
    df['description'] = df.get('description', '').astype(str).fillna('')
    df['category'] = df.get('category', '').astype(str).fillna('')

    df['transaction_type'] = 'Other'  # Default type
    df['category_str'] = df['category'].astype(str)  # Work copy

    # --- Rule 1: Savings transfers ---
    savings_mask = df['description'].str.contains(
        fr'ONLINE TRANSFER TO.*{SAVINGS_NUMBER}$',
        case=False, na=False
    )
    df.loc[savings_mask, 'category'] = 'Savings'
    df.loc[savings_mask, 'transaction_type'] = 'Transfer'

    # --- Rule 2: Income from Paychecks/Other income ---
    income_mask = df['category_str'].str.lower().isin(['paychecks', 'other income'])
    df.loc[income_mask, 'transaction_type'] = 'Income'

    # --- Rule 3: Zelle transfers ---
    transfer_mask = df['category_str'].str.contains("transfer", case=False, na=False) & ~savings_mask

    # Income if withdrawal is blank
    zelle_income_mask = transfer_mask & df['withdrawal'].isna()
    df.loc[zelle_income_mask, ['transaction_type', 'category']] = ['Income', 'Zelle Income']

    # Expense if withdrawal is present
    zelle_expense_mask = transfer_mask & df['withdrawal'].notna()
    df.loc[zelle_expense_mask, ['transaction_type', 'category']] = ['Expense', 'Zelle Expense']

    # --- Fallback: If category matches budget list, treat as Expense ---
    fallback_mask = (df['transaction_type'] == 'Other') & df['category_str'].isin(BUDGET_CATEGORIES)
    df.loc[fallback_mask, 'transaction_type'] = 'Expense'

    # Drop helper col
    df.drop(columns=['category_str'], inplace=True, errors='ignore')
    return df


# ---------- PROCESSING ----------
def process_dataframe(df_raw, column_map):
    """
    Standardizes the raw DataFrame to use consistent column names.
    Also ensures amount is calculated from deposits/withdrawals if missing.
    """
    # Always require description & category
    cols_to_keep = [column_map[k] for k in ['description', 'category'] if column_map.get(k)]

    # Add optional columns if they exist
    for optional_col in ['date', 'amount', 'deposit', 'withdrawal']:
        if column_map.get(optional_col):
            cols_to_keep.append(column_map[optional_col])

    # Subset dataframe to only keep needed columns
    standard_df = df_raw[cols_to_keep].copy()

    # Ensure deposit and withdrawal columns exist (add if missing)
    if 'deposit' not in standard_df.columns:
        standard_df['deposit'] = np.nan
    if 'withdrawal' not in standard_df.columns:
        standard_df['withdrawal'] = np.nan

    # Rename uploaded columns to standard names
    rename_map = {v: k for k, v in column_map.items() if v}
    standard_df.rename(columns=rename_map, inplace=True)

    # Clean numeric columns (amount, deposit, withdrawal)
    for col in ['amount', 'deposit', 'withdrawal']:
        if col in standard_df.columns:
            if standard_df[col].dtype == object:  # If it's string-like
                standard_df[col] = standard_df[col].astype(str).str.replace(r'[$,]', '', regex=True)
            # Convert to numbers
            standard_df[col] = pd.to_numeric(standard_df[col], errors='coerce')

    # Compute "amount" if missing
    def compute_amount(row):
        if 'amount' in row and not pd.isna(row['amount']) and row['amount'] != 0:
            return row['amount']
        if pd.isna(row.get('withdrawal')) or row.get('withdrawal') == 0:
            return -row.get('deposit', 0) if not pd.isna(row.get('deposit')) else 0
        if pd.isna(row.get('deposit')) or row.get('deposit') == 0:
            return row.get('withdrawal', 0) if not pd.isna(row.get('withdrawal')) else 0
        return 0

    standard_df['amount'] = standard_df.apply(compute_amount, axis=1)

    # Parse dates if available
    if 'date' in standard_df:
        standard_df['date'] = pd.to_datetime(standard_df['date'], errors='coerce')
        standard_df['year'] = standard_df['date'].dt.year
        standard_df['month'] = standard_df['date'].dt.month_name()
    else:
        standard_df['year'] = np.nan
        standard_df['month'] = ''

    # Clean category
    standard_df['category'] = standard_df['category'].astype(str).str.strip()

    # Classify transactions
    standard_df = classify_transactions(standard_df)

    # Extract merchant name
    if nlp:
        docs = nlp.pipe(standard_df['description'].astype(str))
        merchants = [
            next((ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]), doc.text.split(' ')[0])
            for doc in docs
        ]
        standard_df['merchant'] = merchants
    else:
        standard_df['merchant'] = standard_df['description'].astype(str).str.split().str[0]

    return standard_df


# ---------- BUDGET ALIGN ----------
def align_budgets_with_categories(df, budget_targets):
    """
    Make sure every category found in the data has a budget entry.
    If a category is missing in the budget_targets, it gets a default of 0.
    """
    categories = df['category'].unique()  # Get all unique categories from the data
    aligned_budget = {cat: budget_targets.get(cat, 0) for cat in categories}

    # Also ensure every "official" budget category exists in the dictionary
    for bc in BUDGET_CATEGORIES:
        if bc not in aligned_budget:
            aligned_budget[bc] = budget_targets.get(bc, 0)
    return aligned_budget


# ---------- RECURRING CHARGES ----------
def find_recurring_charges(df_year):
    """
    Identifies recurring charges.
    Logic:
      - Look only at expenses in "Services" category.
      - Group by merchant + absolute amount.
      - If a transaction happens in at least 3 different months → treat as recurring.
    """
    # Filter to only "Expense" transactions in Services category
    expenses = df_year[
        (df_year['transaction_type'] == 'Expense') &
        (df_year['category'].str.contains("Services", case=False, na=False))
    ].copy()

    if expenses.empty:
        return pd.DataFrame()

    # Normalize amounts (absolute value, rounded)
    expenses['amount_abs'] = expenses['amount'].abs().round(0)

    # Group by merchant + amount, count number of unique months
    recurring_groups = expenses.groupby(['merchant', 'amount_abs'])['month'].nunique()

    # Keep only those that appear in >= 3 months
    potential_recurring = recurring_groups[recurring_groups >= 3].reset_index()

    if not potential_recurring.empty:
        # Also calculate average amount for those recurring transactions
        avg_amount = expenses.groupby(['merchant', 'amount_abs'])['amount'].mean().abs().reset_index(
            name='Average Amount'
        )

        # Merge count + average
        recurring_df = pd.merge(potential_recurring, avg_amount, on=['merchant', 'amount_abs'])
        recurring_df = recurring_df.rename(columns={'month': 'Frequency (Months)'})

        # Return summary table
        return recurring_df[['merchant', 'Average Amount', 'Frequency (Months)']]

    return pd.DataFrame()



# ---------- STYLING ----------
def style_budget_difference(val):
    """
    Style helper for budget DataFrame.
    Colors numbers red if they are negative (overspent),
    green if they are positive (under budget).
    """
    return 'color: red' if val < 0 else 'color: green'


# ---------- VISUALIZATIONS ----------
def plot_pie_chart(category_totals):
    """
    Pie chart of category totals (only positive expenses).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    category_totals = category_totals[category_totals > 0]

    wedges, _, _ = ax.pie(
        category_totals,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.75
    )

    # Add legend
    ax.legend(wedges, category_totals.index, title="Categories",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.axis('equal')  # Equal aspect ratio (circle)
    plt.tight_layout()
    return fig


def plot_bar_chart(category_totals):
    """
    Bar chart of total expenses by category (absolute values).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    category_totals.plot(kind='bar', ax=ax)
    ax.set_ylabel("Total ($)")
    ax.set_title("Expenses by Category (Absolute $)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_normalized_bar(category_totals):
    """
    Bar chart of expenses normalized to % of total.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Normalize to percentages if total > 0
    normalized = category_totals / category_totals.sum() * 100 if category_totals.sum() > 0 else category_totals
    normalized.plot(kind='bar', ax=ax)

    ax.set_ylabel("Percent of Total Expenses (%)")
    ax.set_title("Expenses by Category (Normalized %)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_monthly_trends(df_year):
    """
    Stacked bar chart of monthly spending broken down by category.
    """
    # Pivot: rows = month, columns = category, values = amount
    monthly = df_year[df_year['transaction_type'] == 'Expense'].pivot_table(
        index='month', columns='category',
        values='amount', aggfunc='sum',
        fill_value=0
    )

    # Order months correctly (Jan–Dec)
    try:
        month_order = pd.to_datetime(monthly.index, format='%B').month
        monthly = monthly.iloc[np.argsort(month_order)]
    except Exception:
        pass

    # Plot stacked bars
    fig, ax = plt.subplots(figsize=(11, 6))
    monthly.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Amount ($)")
    ax.set_title("Monthly Spending by Category (stacked)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig, monthly



# ---------- STREAMLIT APP UI ----------
st.title("📊 Financial Analyzer Pro")
st.markdown("Upload your transaction CSV(s). Auto-detect columns, then process data.")

# Session state to persist across reruns
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'file_names' not in st.session_state:
    st.session_state.file_names = None
if 'column_map' not in st.session_state:
    st.session_state.column_map = None

# Upload multiple CSVs
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]

    # If new files uploaded, reset state
    if current_file_names != st.session_state.file_names:
        st.session_state.processed_data = None
        st.session_state.file_names = current_file_names
        st.session_state.column_map = None

    try:
        # Read and clean all CSVs, then combine
        all_dfs = []
        for f in uploaded_files:
            df_temp = pd.read_csv(f, keep_default_na=False)

            # Normalize column names: strip spaces, lowercase
            df_temp.columns = df_temp.columns.str.strip().str.lower()

            # Drop duplicate column names
            df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
            all_dfs.append(df_temp)

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Only process if not already cached
        if st.session_state.processed_data is None:
            # Auto-map columns using aliases
            auto_map = find_and_map_columns(combined_df)

            # Allow user to review/adjust mappings
            st.subheader("Step 1: Verify Column Mapping")
            st.info("We auto-detected your columns. Please confirm or adjust if needed.")

            user_map = {}
            for key in COLUMN_ALIASES.keys():
                # Dropdown options = (blank) + all actual columns
                options = ["(blank)"] + combined_df.columns.tolist()

                # Preselect the detected column (if found)
                default = auto_map.get(key)
                selected = st.selectbox(
                    f"{key.capitalize()} Column",
                    options=options,
                    index=options.index(default) if default in options else 0
                )
                user_map[key] = None if selected == "(blank)" else selected

            # Process data when user confirms
            if st.button("Confirm and Process Data"):
                try:
                    st.session_state.processed_data = process_dataframe(combined_df, user_map)
                    st.session_state.column_map = user_map
                    st.success("Data processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing error: {e}")

    except Exception as e:
        st.error(f"An error occurred reading uploaded files: {e}")


# --- ANALYSIS DISPLAY ---
# ---------- FINAL ANALYSIS / DASHBOARD ----------
if st.session_state.processed_data is not None:
    df = st.session_state.processed_data

    # Let user select which year to analyze
    years = df['year'].dropna().unique()
    years = sorted([int(y) for y in years if not pd.isna(y)])
    selected_year = st.selectbox("Select Year", years)

    if selected_year:
        # Filter data for selected year
        df_year = df[df['year'] == selected_year]

        # ---- Income / Expense / Savings Summary ----
        st.subheader(f"📅 {selected_year} Summary")

        total_income = df_year[df_year['transaction_type'] == 'Income']['amount'].sum()
        total_expense = df_year[df_year['transaction_type'] == 'Expense']['amount'].sum()
        savings = total_income + total_expense  # expenses are negative, so sum = net balance

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Income", f"${total_income:,.2f}")
        col2.metric("Total Expenses", f"${-total_expense:,.2f}")
        col3.metric("Net Savings", f"${savings:,.2f}")

        # ---- Category Totals ----
        st.subheader("📂 Category Breakdown")

        category_totals = df_year.groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        st.dataframe(category_totals.reset_index().rename(columns={'amount': 'Total ($)'}))

        # ---- Recurring Charges ----
        st.subheader("🔁 Recurring Charges (Services)")
        recurring_df = find_recurring_charges(df_year)
        if not recurring_df.empty:
            st.dataframe(recurring_df)
        else:
            st.write("No recurring charges detected.")

        # ---- Monthly Trends ----
        st.subheader("📆 Monthly Spending Trends")
        fig, monthly = plot_monthly_trends(df_year)
        st.pyplot(fig)
        st.dataframe(monthly)

        # ---- Charts ----
        st.subheader("📊 Visualizations")

        # Pie chart
        st.markdown("**Expenses by Category (Pie Chart)**")
        fig = plot_pie_chart(category_totals)
        st.pyplot(fig)

        # Absolute bar chart
        st.markdown("**Expenses by Category (Absolute $)**")
        fig = plot_bar_chart(category_totals)
        st.pyplot(fig)

        # Normalized bar chart
        st.markdown("**Expenses by Category (Normalized %)**")
        fig = plot_normalized_bar(category_totals)
        st.pyplot(fig)
