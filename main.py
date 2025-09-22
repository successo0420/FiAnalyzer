import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import streamlit as st

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide")

# Load spaCy model (optional but used to extract merchant names)
try:
    nlp = spacy.load("en_core_web_sm")
except (OSError, ImportError):
    st.warning(
        "SpaCy model 'en_core_web_sm' not found. Merchant extraction will be basic. To install, run: python -m spacy download en_core_web_sm")
    nlp = None

# Your budget qualifiers (categories)
BUDGET_CATEGORIES = [
    "Merchandise", "Gasoline", "Payments and Credits", "Restaurants", "Supermarkets",
    "Services", "Department Stores", "Education", "Travel/ Entertainment",
    "Awards and Rebate Credits", "Automotive", "Medical Services", "Government Services",
]

# Create a budget dictionary (default 0) — you can fill real budget numbers here
BUDGET_TARGETS = {cat: 0 for cat in BUDGET_CATEGORIES}

# Categories to skip from expense calculations if negative (Payments & Awards)
SKIP_FOR_NEGATIVE = ["Payments and Credits"]
AWARDS_CATEGORY = "Awards and Rebate Credits"

COLUMN_ALIASES = {
    'date': ['trans. date', 'transaction date', 'date', 'posting date'],
    'description': ['description', 'details', 'merchant', 'payee'],
    'category': ['category', 'type', 'classification'],
    'amount': ['amount', 'value', 'price', 'debit/credit', 'debit/credit (amount)'],
    'deposit': ['deposit', 'credit', 'inflow'],
    'withdrawal': ['withdrawal', 'debit', 'outflow']
}

# ---------- HELPERS: column mapping ----------
# --- CONSTANTS ---
SAVINGS_NUMBER = "1234"  # Hardcode last 4 digits of savings account


# ---------- HELPERS: column mapping ----------
def find_and_map_columns(df):
    """Tries to automatically find required columns, including deposits/withdrawals."""

    mapped_cols = {}
    df_cols_lower = {col.lower(): col for col in df.columns}
    for standard_name, aliases in COLUMN_ALIASES.items():
        mapped_cols[standard_name] = None
        for alias in aliases:
            if alias in df_cols_lower:
                mapped_cols[standard_name] = df_cols_lower[alias]
                break
    return mapped_cols


# ---------- CLASSIFICATION ----------
def classify_transactions(df):
    """Classifies transactions and normalizes amounts with custom rules."""
    df = df.copy()

    # Ensure description and category are strings
    df['description'] = df.get('description', '').astype(str).fillna('')
    df['category'] = df.get('category', '').astype(str).fillna('')

    df['transaction_type'] = 'Other'
    df['category_str'] = df['category'].astype(str)

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

    # --- Default Expense fallback ---
    fallback_mask = (df['transaction_type'] == 'Other') & df['category_str'].isin(BUDGET_CATEGORIES)
    df.loc[fallback_mask, 'transaction_type'] = 'Expense'

    df.drop(columns=['category_str'], inplace=True, errors='ignore')
    return df


# ---------- PROCESSING ----------
def process_dataframe(df_raw, column_map):
    """Standardize and process the dataframe with computed amounts if missing."""
    # Always include description & category

    cols_to_keep = [column_map[k] for k in ['description', 'category'] if column_map.get(k)]

    # Optional columns: date, amount, deposit, withdrawal
    for optional_col in ['date', 'amount', 'deposit', 'withdrawal']:
        if column_map.get(optional_col):
            cols_to_keep.append(column_map[optional_col])

    standard_df = df_raw[cols_to_keep].copy()

    # Ensure deposit and withdrawal columns exist
    if 'deposit' not in standard_df.columns:
        standard_df['deposit'] = np.nan
    if 'withdrawal' not in standard_df.columns:
        standard_df['withdrawal'] = np.nan

    # Rename to standard names
    # Rename columns to standard names
    rename_map = {v: k for k, v in column_map.items() if v}
    standard_df.rename(columns=rename_map, inplace=True)

    # Ensure deposit and withdrawal exist
    if 'deposit' not in standard_df.columns:
        standard_df['deposit'] = np.nan
    if 'withdrawal' not in standard_df.columns:
        standard_df['withdrawal'] = np.nan

    # Normalize numeric fields
    for col in ['amount', 'deposit', 'withdrawal']:
        if col in standard_df:
            standard_df[col] = pd.to_numeric(
                standard_df[col].astype(str).str.replace(r'[$,]', '', regex=True),
                errors='coerce'
            )

    # Compute amount if missing
    def compute_amount(row):
        # If amount exists, use it
        if 'amount' in row and not pd.isna(row['amount']) and row['amount'] != 0:
            return row['amount']
        # If withdrawal is empty, use negative deposit (income)
        if pd.isna(row.get('withdrawal')) or row.get('withdrawal') == 0:
            return -row.get('deposit', 0) if not pd.isna(row.get('deposit')) else 0
        # If deposit is empty, use withdrawal (expense)
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

    # Extract merchant
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
    """Ensure all categories in the data have a budget entry."""
    categories = df['category'].unique()
    aligned_budget = {cat: budget_targets.get(cat, 0) for cat in categories}
    for bc in BUDGET_CATEGORIES:
        if bc not in aligned_budget:
            aligned_budget[bc] = budget_targets.get(bc, 0)
    return aligned_budget


# ---------- RECURRING CHARGES ----------
def find_recurring_charges(df_year):
    """Identifies recurring charges from Expenses in Services."""
    expenses = df_year[(df_year['transaction_type'] == 'Expense') & (
        df_year['category'].str.contains("Services", case=False, na=False))].copy()
    if expenses.empty:
        return pd.DataFrame()

    expenses['amount_abs'] = expenses['amount'].abs().round(0)
    recurring_groups = expenses.groupby(['merchant', 'amount_abs'])['month'].nunique()
    potential_recurring = recurring_groups[recurring_groups >= 3].reset_index()
    if not potential_recurring.empty:
        avg_amount = expenses.groupby(['merchant', 'amount_abs'])['amount'].mean().abs().reset_index(
            name='Average Amount')
        recurring_df = pd.merge(potential_recurring, avg_amount, on=['merchant', 'amount_abs'])
        recurring_df = recurring_df.rename(columns={'month': 'Frequency (Months)'})
        return recurring_df[['merchant', 'Average Amount', 'Frequency (Months)']]
    return pd.DataFrame()


# ---------- STYLING ----------
def style_budget_difference(val):
    """Applies color to budget difference."""
    return 'color: red' if val < 0 else 'color: green'


# ---------- VISUALIZATIONS ----------
def plot_pie_chart(category_totals):
    fig, ax = plt.subplots(figsize=(6, 6))
    category_totals = category_totals[category_totals > 0]
    wedges, _, _ = ax.pie(category_totals, autopct='%1.1f%%', startangle=140, pctdistance=0.75)
    ax.legend(wedges, category_totals.index, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.axis('equal')
    plt.tight_layout()
    return fig


def plot_bar_chart(category_totals):
    fig, ax = plt.subplots(figsize=(9, 5))
    category_totals.plot(kind='bar', ax=ax)
    ax.set_ylabel("Total ($)")
    ax.set_title("Expenses by Category (Absolute $)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_normalized_bar(category_totals):
    """Normalized to percent of total expenses."""
    fig, ax = plt.subplots(figsize=(9, 5))
    normalized = category_totals / category_totals.sum() * 100 if category_totals.sum() > 0 else category_totals
    normalized.plot(kind='bar', ax=ax)
    ax.set_ylabel("Percent of Total Expenses (%)")
    ax.set_title("Expenses by Category (Normalized %)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_monthly_trends(df_year):
    """Generates a stacked bar chart of monthly spending."""
    monthly = df_year[df_year['transaction_type'] == 'Expense'].pivot_table(index='month', columns='category',
                                                                            values='amount', aggfunc='sum',
                                                                            fill_value=0)
    try:
        month_order = pd.to_datetime(monthly.index, format='%B').month
        monthly = monthly.iloc[np.argsort(month_order)]
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(11, 6))
    monthly.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Amount ($)")
    ax.set_title("Monthly Spending by Category (stacked)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig, monthly


# ---------- STREAMLIT APP UI ----------
st.title("📊 Financial Analyzer Pro")
st.markdown("Upload your transaction CSV(s). Map columns, then press **Confirm and Process Data**.")

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'file_names' not in st.session_state:
    st.session_state.file_names = None
if 'column_map' not in st.session_state:
    st.session_state.column_map = None

uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    if current_file_names != st.session_state.file_names:
        st.session_state.processed_data = None
        st.session_state.file_names = current_file_names
        st.session_state.column_map = None

    try:
        # Read and combine all CSVs
        all_dfs = []
        for f in uploaded_files:
            df_temp = pd.read_csv(f, keep_default_na=False)
            # Strip spaces, lowercase, remove duplicate columns
            df_temp.columns = df_temp.columns.str.strip().str.lower()
            df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
            all_dfs.append(df_temp)
        combined_df = pd.concat(all_dfs, ignore_index=True)

        if st.session_state.processed_data is None:
            # --- COLUMN MAPPING WITH (blank) OPTION ---
            st.subheader("Step 1: Map Your Columns")
            st.info(
                "Select the correct column from your CSV for each field. Use '(blank)' if the column does not exist.")

            user_map = {}
            for key in COLUMN_ALIASES.keys():
                # Previous default if available
                default_col = st.session_state.column_map.get(key) if st.session_state.column_map else None

                # Add "(blank)" at top of options
                options = ["(blank)"] + combined_df.columns.tolist()

                # Dropdown selectbox
                selected = st.selectbox(
                    f"{key.capitalize()} Column",
                    options=options,
                    index=options.index(default_col) if default_col in options else 0
                )

                # Convert "(blank)" to None
                user_map[key] = None if selected == "(blank)" else selected

            # Process button
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
if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
    df = st.session_state.processed_data.copy()
    st.success("Data processed successfully! Here is your analysis.")
    st.markdown("---")

    years = sorted(df['year'].dropna().unique().astype(int), reverse=True)
    if not years:
        st.warning("No valid dates detected after processing; ensure your CSV has a valid date column.")
    else:
        selected_year = st.selectbox('**Select a Year to Analyze**', options=years)
        df_year = df[df['year'] == int(selected_year)]

        st.header(f"Financial Summary for {selected_year}")

        total_income = df_year.loc[df_year['transaction_type'] == 'Income', 'amount'].sum()
        total_expenses = df_year.loc[df_year['transaction_type'] == 'Expense', 'amount'].sum()
        net_savings = total_income - total_expenses

        # --- MODIFIED: Changed to 3 columns ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Income", f"${total_income:,.2f}", help="Sum of transactions classified as 'Income'.")
        col2.metric("Total Expenses", f"${total_expenses:,.2f}", help="Sum of transactions classified as 'Expense'.")
        col3.metric("Net Savings", f"${net_savings:,.2f}", delta=f"{net_savings:,.2f}")

        st.markdown("---")
        st.subheader("Total Yearly Spending by Category (Normalized %)")
        yearly_expenses_by_cat = df_year[df_year['transaction_type'] == 'Expense'].groupby('category')['amount'].sum()
        if yearly_expenses_by_cat.sum() > 0:
            st.bar_chart((yearly_expenses_by_cat / yearly_expenses_by_cat.sum() * 100).sort_values(ascending=False))
        else:
            st.info("No expense data for the selected year.")

        st.markdown("---")
        st.header(f"Monthly Breakdown for {selected_year}")
        months_with_data = sorted(df_year['month'].unique(), key=lambda m: pd.to_datetime(m, format='%B').month)
        if months_with_data:
            selected_month = st.selectbox('**Select a Month for a Detailed Summary**', options=months_with_data)
        else:
            selected_month = None
            st.info("No monthly data available.")

        if selected_month:
            df_month = df_year[df_year['month'] == selected_month]
            col1_month, col2_month = st.columns(2)
            with col1_month:
                st.subheader(f"Spending in {selected_month}")
                category_totals = df_month[df_month['transaction_type'] == 'Expense'].groupby('category')[
                    'amount'].sum()
                if not category_totals.empty and category_totals.sum() > 0:
                    st.pyplot(plot_pie_chart(category_totals))
                else:
                    st.info(f"No spending data for {selected_month}.")
            with col2_month:
                st.subheader("Budget vs. Actual")
                aligned_budgets = align_budgets_with_categories(df_year, BUDGET_TARGETS)
                budget_data = []
                for c, b in aligned_budgets.items():
                    actual = float(category_totals.get(c,
                                                       0)) if 'category_totals' in locals() and not category_totals.empty else 0.0
                    budget_data.append({"Category": c, "Budget": b, "Actual": actual, "Difference": b - actual})
                budget_df = pd.DataFrame(budget_data).set_index('Category')
                st.dataframe(
                    budget_df.style.apply(lambda x: x.map(style_budget_difference), subset=['Difference']).format(
                        '${:,.2f}'))

        st.markdown("---")
        st.header("Recurring Charges Analysis (Services Only)")
        recurring_df = find_recurring_charges(df_year)
        if not recurring_df.empty:
            st.write(recurring_df.style.format({'Average Amount': '${:,.2f}'}))
        else:
            st.info("No recurring charges detected in 'Services' for this year.")

        st.markdown("---")
        # --- MODIFIED: Charts panel now behind a button ---
        st.header("Spending Visualizations")
        if st.button("Show Spending Charts"):
            chart_category_totals = df_year[df_year['transaction_type'] == 'Expense'].groupby('category')[
                'amount'].sum().reindex(BUDGET_CATEGORIES, fill_value=0)
            if chart_category_totals.sum() > 0:
                st.subheader("Absolute Expenses by Category")
                st.pyplot(plot_bar_chart(chart_category_totals))

                st.subheader("Normalized Expenses by Category (%)")
                st.pyplot(plot_normalized_bar(chart_category_totals))

                st.subheader("Monthly Stacked Trend (Expenses)")
                fig_monthly, monthly_df = plot_monthly_trends(df_year)
                st.pyplot(fig_monthly)
                st.dataframe(monthly_df)
            else:
                st.info("No expense data to plot for the selected year.")