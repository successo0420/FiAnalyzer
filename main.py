import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, TableStyle, Table

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="Financial Analyzer Pro")

# Try to load the SpaCy NLP model (used for merchant name extraction)
# If it's not installed, fallback to basic string handling.

# Predefined budget categories
BUDGET_CATEGORIES = [
    "Merchandise", "Gasoline", "Payments and Credits", "Restaurants", "Supermarkets",
    "Services", "Department Stores", "Education", "Travel/ Entertainment",
    "Awards and Rebate Credits", "Automotive", "Medical Services", "Government Services",
    "Utilities", "Personal Expenses", "Credit Card Payment", "Transfers", "Groceries",
    "Other Income", "Restaurants and Dining", "Deposits", "Office Supplies", "Paychecks",
    "General Merchandise", "Services and Supplies", "Travel"
]

# Category mapping to merge similar categories
CATEGORY_MAPPINGS = {
    # Merge similar restaurant categories
    "Restaurants and Dining": "Restaurants",

    # Merge similar grocery categories
    "Groceries": "Supermarkets",

    # Merge similar merchandise categories
    "General Merchandise": "Merchandise",

    # Merge similar services categories
    "Services and Supplies": "Services",

    # Merge similar travel categories
    "Travel": "Travel/ Entertainment",

    # You can add more mappings here as needed
    # "Original Category": "Target Category"
}

# Default budget targets for each category (all 0 for now, can be filled later)
BUDGET_TARGETS = {cat: 100 for cat in BUDGET_CATEGORIES}

# Categories that shouldn't be treated as negative expenses
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
    Classifies each transaction into Income, Expense, Transfer, Credit Card Payment, etc.
    Also normalizes some categories based on rules.
    """
    df = df.copy()

    # Make sure description and category are strings (avoid dtype errors)
    if 'description' in df.columns:
        df['description'] = df['description'].astype(str).fillna('')
    else:
        df['description'] = ''

    if 'category' in df.columns:
        df['category'] = df['category'].astype(str).fillna('')
    else:
        df['category'] = ''

    df['transaction_type'] = 'Other'  # Default type
    df['category_str'] = df['category'].astype(str)  # Work copy

    # --- Rule 1: Credit Card Payments take priority ---
    cc_keywords = ['credit card', 'cc payment', 'card payment', 'payment to', 'autopay']
    cc_mask = df['category_str'].str.contains('|'.join(cc_keywords), case=False, na=False)

    df.loc[cc_mask, 'category'] = 'Credit Card Payment'
    df.loc[cc_mask, 'transaction_type'] = 'Credit Card Payment'

    # --- Rule 2: If not CC payment, handle Payments and Credits ---
    payments_credits_mask = (df['category_str'] == 'Payments and Credits') & ~cc_mask
    df.loc[payments_credits_mask, 'transaction_type'] = 'Duplicate CC Payment'
    df.loc[payments_credits_mask, 'category'] = 'Payments and Credits'

    # --- Rule 4: Income from Paychecks/Other income/Awards ---
    income_categories = ['paychecks', 'other income', 'awards and rebate credits']
    income_mask = df['category_str'].str.lower().isin(income_categories)
    df.loc[income_mask, 'transaction_type'] = 'Income'

    # Ensure Awards and Rebates are properly categorized as income
    awards_mask = df['category_str'].str.contains('award|rebate', case=False, na=False)
    df.loc[awards_mask, 'transaction_type'] = 'Income'
    df.loc[awards_mask, 'category'] = AWARDS_CATEGORY

    # --- Rule 5a: Account-to-account transfers (highest priority for transfers) ---
    account_transfer_mask = df['description'].str.contains('ONLINE TRANSFER TO', case=False, na=False)
    df.loc[account_transfer_mask, 'transaction_type'] = 'Transfer'
    df.loc[account_transfer_mask, 'category'] = 'Account Transfer'

    # --- Rule 5b: Zelle/external person-to-person transfers ---
    external_transfer_keywords = ['zelle', 'zel', 'venmo', 'paypal', 'cashapp', 'apple pay']
    zelle_mask = (df['category_str'].str.contains('|'.join(external_transfer_keywords), case=False, na=False) &
                  ~account_transfer_mask)  # Exclude account transfers

    # Use deposit/withdrawal columns to determine if Zelle is income or expense
    if 'deposit' in df.columns and 'withdrawal' in df.columns:
        # If deposit has value and withdrawal is NaN/blank, it's incoming Zelle (income)
        incoming_zelle = zelle_mask & df['deposit'].notna() & df['withdrawal'].isna()
        df.loc[incoming_zelle, 'transaction_type'] = 'Transfer'
        df.loc[incoming_zelle, 'category'] = 'Zelle Income'

        # If withdrawal has value and deposit is NaN/blank, it's outgoing Zelle (expense)
        outgoing_zelle = zelle_mask & df['withdrawal'].notna() & df['deposit'].isna()
        df.loc[outgoing_zelle, 'transaction_type'] = 'Transfer'
        df.loc[outgoing_zelle, 'category'] = 'Zelle Expense'
    else:
        # Fallback if deposit/withdrawal columns don't exist
        df.loc[zelle_mask, 'transaction_type'] = 'Transfer'
        df.loc[zelle_mask, 'category'] = 'Zelle Transfer'

    # --- Rule 5c: Other generic transfers ---
    other_transfer_keywords = ['transfer', 'mobile deposit', 'wire transfer']
    other_transfer_mask = (df['category_str'].str.contains('|'.join(other_transfer_keywords), case=False, na=False) &
                           ~account_transfer_mask & ~zelle_mask)  # Exclude already classified transfers

    df.loc[other_transfer_mask, 'transaction_type'] = 'Transfer'

    # --- Rule 6: Income detection from deposits ---
    income_keywords = ['deposit', 'payroll', 'salary', 'direct deposit', 'interest paid']
    income_desc_mask = df['description'].str.contains('|'.join(income_keywords), case=False, na=False)

    # Check for deposits or positive amounts
    if 'deposit' in df.columns:
        deposit_income_mask = income_desc_mask & df['deposit'].notna()
    else:
        deposit_income_mask = income_desc_mask & (df['amount'] > 0)

    unclassified_mask = df['transaction_type'] == 'Other'
    df.loc[deposit_income_mask & unclassified_mask, 'transaction_type'] = 'Income'

    # --- Rule 7: Expense detection ---
    # Expenses should be positive numbers - if we have negative amounts, they need to be handled
    if 'withdrawal' in df.columns:
        # Use withdrawal column for expense detection
        withdrawal_expense_mask = df['withdrawal'].notna()
        df.loc[withdrawal_expense_mask & unclassified_mask, 'transaction_type'] = 'Expense'
    # else:
    #     # Fallback: look for negative amounts (but remember expenses should be positive)
    #     negative_amount_mask = df['amount'] < 0
    #     df.loc[negative_amount_mask & unclassified_mask, 'transaction_type'] = 'Expense'

    # Also check for expense keywords
    expense_keywords = ['purchase', 'payment', 'fee', 'charge', 'debit']
    expense_desc_mask = df['description'].str.contains('|'.join(expense_keywords), case=False, na=False)
    df.loc[expense_desc_mask & unclassified_mask, 'transaction_type'] = 'Expense'

    # --- Fallback: If category matches budget list, treat as Expense ---
    fallback_mask = (df['transaction_type'] == 'Other') & df['category_str'].isin(BUDGET_CATEGORIES)
    df.loc[fallback_mask, 'transaction_type'] = 'Expense'

    # --- Final cleanup: Amount normalization ---
    # Ensure expenses are positive numbers
    expense_mask = df['transaction_type'] == 'Expense'
    df.loc[expense_mask, 'amount'] = df.loc[expense_mask, 'amount'].abs()

    # Ensure income are positive numbers
    income_mask = df['transaction_type'] == 'Income'
    df.loc[income_mask, 'amount'] = df.loc[income_mask, 'amount'].abs()

    # For credit card payments and transfers, we typically want positive amounts for analysis
    cc_payment_mask = df['transaction_type'] == 'Credit Card Payment'
    df.loc[cc_payment_mask, 'amount'] = df.loc[cc_payment_mask, 'amount'].abs()

    transfer_mask = df['transaction_type'] == 'Transfer'
    df.loc[transfer_mask, 'amount'] = df.loc[transfer_mask, 'amount'].abs()

    # Drop helper col
    df.drop(columns=['category_str'], inplace=True, errors='ignore')

    return df


def smart_category_mapping(categories):
    """
    Use NLP similarity to automatically group similar categories.
    Only returns mappings that need user review (high confidence matches).
    """
    from difflib import SequenceMatcher

    # Predefined smart mappings based on common patterns
    smart_mappings = {}
    auto_mappings = {}

    # Define keyword groups for automatic mapping
    keyword_groups = {
        "Restaurants": ["restaurant", "dining", "food", "eat", "cafe", "coffee", "pizza", "burger"],
        "Supermarkets": ["grocery", "groceries", "supermarket", "market", "food store"],
        "Gasoline": ["gas", "fuel", "gasoline", "petrol", "shell", "exxon", "bp"],
        "Services": ["service", "services", "repair", "maintenance", "professional"],
        "Merchandise": ["merchandise", "general merchandise", "retail", "shopping"],
        "Travel/ Entertainment": ["travel", "entertainment", "hotel", "flight", "movie", "theater"],
        "Automotive": ["auto", "automotive", "vehicle", "mechanic"],
        "Medical Services": ["medical", "health", "doctor", "hospital", "pharmacy", "dental"],
        "Utilities": ["utility", "utilities", "electric", "water", "gas", "internet", "phone"]
    }

    for category in categories:
        if not category or category == '':
            continue

        category_lower = category.lower().strip()
        best_match = None
        best_score = 0

        # Check keyword matching first
        for target_category, keywords in keyword_groups.items():
            for keyword in keywords:
                if keyword in category_lower:
                    if len(keyword) > best_score:  # Longer matches are better
                        best_match = target_category
                        best_score = len(keyword)

        # If we found a good keyword match, auto-map it
        if best_match and best_score >= 4:  # Minimum keyword length
            auto_mappings[category] = best_match
        else:
            # Check for exact or near-exact matches with budget categories
            for budget_cat in BUDGET_CATEGORIES:
                similarity = SequenceMatcher(None, category_lower, budget_cat.lower()).ratio()
                if similarity > 0.8:  # High similarity threshold
                    if similarity > best_score:
                        best_match = budget_cat
                        best_score = similarity

            # If similarity is very high, auto-map; otherwise, suggest for review
            if best_score > 0.9:
                auto_mappings[category] = best_match
            elif best_score > 0.6:
                smart_mappings[category] = best_match

    return auto_mappings, smart_mappings


def extract_account_numbers(df):
    """
    Extract account numbers from transfer descriptions.
    Returns a set of unique account numbers found.
    """
    import re
    account_numbers = set()

    if 'description' not in df.columns:
        return account_numbers

    # Look for patterns like "XXXXX1234" or "TO XXXXX1234"
    pattern = r'ONLINE TRANSFER TO\s+\w{5}(\d{4})'

    for desc in df['description'].astype(str):
        matches = re.findall(pattern, desc.upper())
        account_numbers.update(matches)

    return sorted(list(account_numbers))


def calculate_account_balances(df, account_mappings):
    """
    Calculate the net amount for each account based on transfers.
    """
    if 'description' not in df.columns or not account_mappings:
        return {}

    # Initialize balances for all mapped accounts
    account_balances = {
        f"{account_type} (ending {account_num})": 0
        for account_num, account_type in account_mappings.items()
    }

    # Pattern to find the last 4 digits of an account number
    account_pattern = r'(\d{4})'

    # Filter for relevant transaction types
    transfer_df = df[df['transaction_type'].isin(['Transfer', 'Credit Card Payment'])].copy()

    for _, row in transfer_df.iterrows():
        desc = str(row['description']).upper()
        amount = row.get('amount', 0)

        matches = re.findall(account_pattern, desc)
        for account_num in matches:
            if account_num in account_mappings:
                account_type = account_mappings[account_num]
                account_key = f"{account_type} (ending {account_num})"

                # If "TO" is in the description, it's an inflow for that account.
                # Otherwise, assume it's an outflow.
                if 'TO' in desc:
                    account_balances[account_key] += abs(amount)
                else:
                    account_balances[account_key] -= abs(amount)
                break  # Process only the first matched account number in a description

    return account_balances


def apply_category_mappings(df, mappings):
    """
    Apply category mappings to merge similar categories.
    """
    df = df.copy()
    df['category'] = df['category'].replace(mappings)
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

    # Remove None values and duplicates
    cols_to_keep = list(set([col for col in cols_to_keep if col is not None]))

    # Subset dataframe to only keep needed columns
    standard_df = df_raw[cols_to_keep].copy()

    # Rename uploaded columns to standard names
    rename_map = {v: k for k, v in column_map.items() if v and v in standard_df.columns}
    standard_df.rename(columns=rename_map, inplace=True)

    # Ensure required columns exist
    required_cols = ['description', 'category']
    for col in required_cols:
        if col not in standard_df.columns:
            standard_df[col] = ''

    # Ensure deposit and withdrawal columns exist (add if missing)
    if 'deposit' not in standard_df.columns:
        standard_df['deposit'] = np.nan
    if 'withdrawal' not in standard_df.columns:
        standard_df['withdrawal'] = np.nan

    # Clean numeric columns (amount, deposit, withdrawal)
    for col in ['amount', 'deposit', 'withdrawal']:
        if col in standard_df.columns:
            # Convert to string first, then clean
            standard_df[col] = standard_df[col].astype(str)
            # Remove currency symbols and commas
            standard_df[col] = standard_df[col].str.replace(r'[$,]', '', regex=True)
            # Convert to numeric
            standard_df[col] = pd.to_numeric(standard_df[col], errors='coerce')

    # Compute "amount" if missing
    def compute_amount(row):
        if 'amount' in row and not pd.isna(row['amount']) and row['amount'] != 0:
            return row['amount']

        deposit_val = row.get('deposit', 0)
        withdrawal_val = row.get('withdrawal', 0)

        if pd.isna(deposit_val):
            deposit_val = 0
        if pd.isna(withdrawal_val):
            withdrawal_val = 0

        if withdrawal_val == 0:
            return -deposit_val
        elif deposit_val == 0:
            return withdrawal_val
        else:
            return 0

    standard_df['amount'] = standard_df.apply(compute_amount, axis=1)

    # Parse dates if available
    if 'date' in standard_df.columns:
        standard_df['date'] = pd.to_datetime(standard_df['date'], errors='coerce')
        standard_df['year'] = standard_df['date'].dt.year
        standard_df['month'] = standard_df['date'].dt.month_name()
        standard_df['month_num'] = standard_df['date'].dt.month  # For sorting
    else:
        standard_df['year'] = np.nan
        standard_df['month'] = ''
        standard_df['month_num'] = np.nan

    # Clean category
    standard_df['category'] = standard_df['category'].astype(str).str.strip()

    # Classify transactions
    standard_df = classify_transactions(standard_df)

    # Apply category mappings to merge similar categories
    standard_df = apply_category_mappings(standard_df, CATEGORY_MAPPINGS)

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
    recurring_categories = {"Services", "Utilities", "Personal Expenses", "Office Supplies"}
    # Filter to only "Expense" transactions in Services category
    expenses = df_year[
        (df_year['transaction_type'] == 'Expense') &
        (df_year['category'].str.contains('|'.join(recurring_categories), case=False, na=False))
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
    fig, ax = plt.subplots(figsize=(10, 8))
    category_totals = category_totals[category_totals > 0]

    if len(category_totals) == 0:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Only show categories that are at least 2% of total to avoid clutter
    total = category_totals.sum()
    significant_categories = category_totals[category_totals / total >= 0.02]
    other_amount = category_totals[category_totals / total < 0.02].sum()

    if other_amount > 0:
        significant_categories = pd.concat([significant_categories, pd.Series({'Other': other_amount})])

    wedges, texts, autotexts = ax.pie(
        significant_categories,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85
    )

    # Add legend
    ax.legend(wedges, significant_categories.index, title="Categories",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title("Expenses by Category Distribution")
    ax.axis('equal')  # Equal aspect ratio (circle)
    plt.tight_layout()
    return fig


def categorize_transfers_by_account(df, account_mappings):
    """
    Categorize transfers based on account number mappings.
    """
    df = df.copy()
    import re

    if 'description' not in df.columns or not account_mappings:
        return df

    # Create pattern to match any account number
    account_pattern = r'[A-Z\s]*(\d{4})(?:\s|$)'

    for idx, row in df.iterrows():
        if row.get('transaction_type') == 'Transfer' or 'transfer' in str(row.get('category', '')).lower():
            desc = str(row['description']).upper()
            matches = re.findall(account_pattern, desc)

            for account_num in matches:
                if account_num in account_mappings:
                    account_type = account_mappings[account_num]
                    # Determine if it's TO or FROM this account
                    if 'to' in desc.lower() and account_num in desc:
                        df.at[idx, 'category'] = f'Transfer to {account_type}'
                    else:
                        df.at[idx, 'category'] = f'Transfer from {account_type}'
                    break

    return df


def plot_bar_chart(category_totals):
    """
    Bar chart of total expenses by category (absolute values).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    if len(category_totals) > 0:
        # Sort by value descending
        category_totals_sorted = category_totals.sort_values(ascending=True)
        bars = ax.barh(range(len(category_totals_sorted)), category_totals_sorted.values)
        ax.set_yticks(range(len(category_totals_sorted)))
        ax.set_yticklabels(category_totals_sorted.index)
        ax.set_xlabel("Total Amount ($)")
        ax.set_title("Expenses by Category")

        # Add value labels on bars
        for i, (idx, val) in enumerate(category_totals_sorted.items()):
            ax.text(val + max(category_totals_sorted) * 0.01, i, f'${val:,.0f}',
                    va='center', fontsize=9)

        plt.subplots_adjust(left=0.2)
    else:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    return fig


def plot_normalized_bar(category_totals):
    """
    Bar chart of expenses normalized to % of total.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if len(category_totals) > 0 and category_totals.sum() > 0:
        # Normalize to percentages if total > 0
        normalized = (category_totals / category_totals.sum() * 100).sort_values(ascending=True)
        bars = ax.barh(range(len(normalized)), normalized.values)
        ax.set_yticks(range(len(normalized)))
        ax.set_yticklabels(normalized.index)
        ax.set_xlabel("Percent of Total Expenses (%)")
        ax.set_title("Expenses by Category (Percentage Distribution)")

        # Add percentage labels on bars
        for i, (idx, val) in enumerate(normalized.items()):
            ax.text(val + max(normalized) * 0.01, i, f'{val:.1f}%',
                    va='center', fontsize=9)

        plt.subplots_adjust(left=0.2)
    else:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    return fig


def plot_monthly_trends(df_year):
    """
    Stacked bar chart of monthly spending broken down by category.
    """
    expenses_df = df_year[df_year['transaction_type'] == 'Expense']

    if expenses_df.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.text(0.5, 0.5, 'No expense data to display', ha='center', va='center', transform=ax.transAxes)
        return fig, pd.DataFrame()

    # Pivot: rows = month, columns = category, values = amount
    monthly = expenses_df.pivot_table(
        index='month', columns='category',
        values='amount', aggfunc='sum',
        fill_value=0
    )

    # Order months correctly (Jan–Dec)
    try:
        if not monthly.empty:
            month_order = pd.to_datetime(monthly.index, format='%B').month
            monthly = monthly.iloc[np.argsort(month_order)]
    except Exception:
        pass

    # Plot stacked bars
    fig, ax = plt.subplots(figsize=(11, 6))
    if not monthly.empty:
        monthly.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel("Amount ($)")
        ax.set_title("Monthly Spending by Category (stacked)")
        plt.xticks(rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    return fig, monthly


# ---------- EXPORT FUNCTIONS ----------
def export_to_excel(df):
    """
    Export dataframe to Excel:
      - First sheet = all transactions
      - Separate sheet for each month (if data exists)
    """
    cols_to_drop = ["deposit", "withdrawal", "transaction_type", "month_num"]

    # Convert df to list of lists for reportlab Table
    df = df.drop(columns=cols_to_drop, errors="ignore")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # --- Full yearly data ---
        df.to_excel(writer, sheet_name="All_Transactions", index=False)

        # --- Monthly sheets ---
        if "date" in df.columns:
            # Make sure date is datetime
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # Drop rows without valid date
            df_valid = df.dropna(subset=["date"]).copy()

            # Group by month
            df_valid["month"] = df_valid["date"].dt.month

            # Write each month as separate sheet
            month_map = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December"
            }

            for m, g in df_valid.groupby("month"):
                sheet_name = month_map.get(m, f"Month{m}")
                g.drop(columns=["month"], inplace=True)
                g.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    return output


def export_to_pdf(df):
    """
    Export processed dataframe to a PDF file (in-memory),
    giving more space to Description and Merchant columns.
    """
    output = BytesIO()

    # Create document with smaller margins
    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(letter),
        leftMargin=20,
        rightMargin=20,
        topMargin=20,
        bottomMargin=20
    )

    elements = []

    # Drop columns you don't want in the report
    cols_to_drop = ["deposit", "withdrawal", "transaction_type", "month_num"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Define column weights
    # Description 3, Merchant 2, others 1 each
    weights = []
    for col in df.columns:
        if col.lower() == "description":
            weights.append(4)
        else:
            weights.append(1)

    # Compute column widths proportional to page width
    page_width, page_height = landscape(letter)
    available_width = page_width - doc.leftMargin - doc.rightMargin
    col_widths = [available_width * w / sum(weights) for w in weights]

    # Convert dataframe to list of lists
    data = [df.columns.tolist()] + df.astype(str).values.tolist()

    # Create table
    table = Table(data, colWidths=col_widths, repeatRows=1)

    # Add styling
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
    ])

    table.setStyle(style)

    elements.append(table)
    doc.build(elements)
    output.seek(0)

    return output


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
if 'account_mappings' not in st.session_state:
    st.session_state.account_mappings = {}

# Upload multiple CSVs
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
if "comp" not in st.session_state:
    st.session_state.comp = pd.DataFrame()

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

            # Get unique categories from the data for mapping
            temp_df = process_dataframe(combined_df, user_map)
            unique_categories = sorted([cat for cat in temp_df['category'].unique() if cat and cat != ''])

            # Extract account numbers for transfer categorization
            account_numbers = extract_account_numbers(temp_df)
            account_mappings = {}

            if account_numbers:
                st.subheader("Step 2a: Map Account Numbers")
                st.info(
                    "We found account numbers in your transfer descriptions. Please specify what type of account each one is:")

                account_types = ["Savings", "Growth", "Investment", "Credit Card", "Other"]

                col1, col2 = st.columns(2)
                for i, account_num in enumerate(account_numbers):
                    with col1 if i % 2 == 0 else col2:
                        selected_type = st.selectbox(
                            f"Account ending in {account_num} is a:",
                            options=account_types,
                            key=f"account_{account_num}"
                        )
                        account_mappings[account_num] = selected_type

            # Apply account-based transfer categorization
            if account_mappings:
                temp_df = categorize_transfers_by_account(temp_df, account_mappings)
            unique_categories = sorted([cat for cat in temp_df['category'].unique() if cat and cat != ''])

            # Category mapping with NLP assistance
            st.subheader("Step 2b: Smart Category Mapping")
            temp_df = process_dataframe(combined_df, user_map)

            # Use NLP to automatically group categories
            auto_mappings, suggested_mappings = smart_category_mapping(unique_categories)

            # Show automatic mappings
            if auto_mappings:
                st.success(f"✅ Automatically mapped {len(auto_mappings)} categories:")
                auto_df = pd.DataFrame([
                    {"Original Category": orig, "Mapped To": target}
                    for orig, target in auto_mappings.items()
                ])
                st.dataframe(auto_df, hide_index=True)

            # Show suggested mappings for user review
            final_mappings = auto_mappings.copy()
            # Update the global mappings
            CATEGORY_MAPPINGS.update(final_mappings)

            # -------------------------
            # 📑 Budget Planner (fixed)
            # -------------------------
            st.subheader("📑 Budget Planner")

            # Build a preview dataframe to compute categories & actuals for the budget template.
            # Use the same process_dataframe logic but apply the final_mappings so the template matches
            # what the processed data will look like once the user hits "Confirm and Process Data".
            try:
                preview_df = process_dataframe(combined_df, user_map)
                if final_mappings:
                    preview_df = apply_category_mappings(preview_df, final_mappings)
            except Exception as e:
                st.error(f"Could not build preview for budget template: {e}")
                preview_df = None

            if preview_df is None or preview_df.empty:
                st.info("No preview data available for budget template yet.")
            else:
                # Build list of categories to include in the template (sorted, non-empty)
                # Filter preview_df to only expenses
                expense_df = preview_df[preview_df['transaction_type'] == 'Expense']

                budget_categories = sorted(
                    [c for c in expense_df['category'].unique() if c and str(c).strip() != ""]
                )

                # Create template DataFrame
                template_df = pd.DataFrame({
                    "Category": budget_categories,
                    "Budget": [""] * len(budget_categories)  # empty cells for user to fill
                })

                # Create an in-memory Excel file (BytesIO) and write template_df to it
                from io import BytesIO

                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    template_df.to_excel(writer, index=False, sheet_name="Budgets")
                buf.seek(0)

                # Download button for the template
                st.download_button(
                    label="⬇️ Download Budget Template (Excel)",
                    data=buf.getvalue(),
                    file_name="budget_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.caption("Download this template, fill the 'Budget' column, then re-upload below.")

                # Upload completed budget file
                uploaded_budget = st.file_uploader("Upload your completed budget file (Excel or CSV)",
                                                   type=["xlsx", "csv"], key="uploaded_budget")
                if uploaded_budget is not None:
                    try:
                        # Read uploaded budget file into DataFrame
                        if uploaded_budget.name.lower().endswith(".xlsx"):
                            budget_df = pd.read_excel(uploaded_budget)
                        else:
                            budget_df = pd.read_csv(uploaded_budget)

                        # normalize header names (strip + lowercase)
                        budget_df.columns = [c.strip() for c in budget_df.columns.astype(str)]

                        # Attempt to detect category & budget columns (case-insensitive)
                        lower_cols = [c.lower() for c in budget_df.columns]
                        # search for suitable category column
                        cat_col = None
                        for candidate in ["category", "cat", "categories", "category name"]:
                            if candidate in lower_cols:
                                cat_col = budget_df.columns[lower_cols.index(candidate)]
                                break
                        if cat_col is None:
                            # fallback to first column if nothing obvious
                            cat_col = budget_df.columns[0]

                        # search for suitable budget column
                        budget_col = None
                        for candidate in ["budget", "amount", "budget ($)", "budget_amount", "value"]:
                            if candidate in lower_cols:
                                budget_col = budget_df.columns[lower_cols.index(candidate)]
                                break
                        if budget_col is None:
                            # fallback to second column if it exists, else set 0 later
                            budget_col = budget_df.columns[1] if len(budget_df.columns) > 1 else None

                        # Prepare cleaned budget_df
                        cleaned = pd.DataFrame()
                        cleaned['Category'] = budget_df[cat_col].astype(str).str.strip()
                        if budget_col is not None:
                            cleaned['Budget'] = pd.to_numeric(budget_df[budget_col], errors='coerce').fillna(0)
                        else:
                            cleaned['Budget'] = 0.0

                        # Build actuals from preview_df (abs amounts, grouped by category)
                        actuals = preview_df.groupby('category')['amount'].sum().abs()
                        actuals = actuals.reindex(budget_categories).fillna(0)  # keep ordering

                        # Build comparison DataFrame (keep numeric columns for export)
                        comp = pd.DataFrame({
                            "Category": budget_categories,
                            "Actual": actuals.values
                        })

                        # Map budgets to categories using case-insensitive match
                        budget_map = dict(zip(cleaned['Category'].str.lower(), cleaned['Budget']))
                        comp['Budget'] = comp['Category'].str.lower().map(budget_map).fillna(0).astype(float)

                        # Calculate difference and status
                        comp['Diff'] = comp['Budget'] - comp['Actual']


                        def status_text(x):
                            if abs(x) < 1e-9:
                                return "✓"
                            return "Under" if x > 0 else "Over"
                        st.session_state.comp = comp


                    except Exception as e:
                        st.error(f"Error reading budget file or computing comparison: {e}")

            # Process data when user confirms
            if st.button("Confirm and Process Data"):
                try:
                    processed_df = process_dataframe(combined_df, user_map)
                    # Apply account-based transfer categorization
                    if account_mappings:
                        processed_df = categorize_transfers_by_account(processed_df, account_mappings)
                    # Apply final category mappings
                    processed_df = apply_category_mappings(processed_df, final_mappings)

                    st.session_state.processed_data = processed_df
                    st.session_state.column_map = user_map
                    st.session_state.account_mappings = account_mappings
                    st.success("Data processed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing error: {e}")

    except Exception as e:
        st.error(f"An error occurred reading uploaded files: {e}")

# --- ANALYSIS DISPLAY ---
if st.session_state.processed_data is not None:
    df = st.session_state.processed_data

    # Let user select which year to analyze
    years = df['year'].dropna().unique()
    years = sorted([int(y) for y in years if not pd.isna(y)])

    if len(years) > 0:
        selected_year = st.selectbox("Select Year", years)

        if selected_year:
            # Filter data for selected year
            df_year = df[df['year'] == selected_year]
            category_totals = df_year.groupby('category')['amount'].sum().abs().sort_values(ascending=False)

            # ---- TABLES SECTION ----
            st.header("📊 Data Tables")

            # ---- MAIN SUMMARY WITH 5 CATEGORIES ----
            st.subheader(f"📅 {selected_year} Financial Summary")

            total_income = df_year[df_year['transaction_type'] == 'Income']['amount'].sum()
            total_expense = df_year[df_year['transaction_type'] == 'Expense']['amount'].sum()
            total_cc_payments = df_year[df_year['transaction_type'] == 'Credit Card Payment']['amount'].sum()
            total_transfers = df_year[df_year['transaction_type'] == 'Transfer']['amount'].sum()
            net_savings = total_income - total_expense  # All should sum to net savings

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Income", f"${total_income:,.2f}")
            col2.metric("Total Expenses", f"${total_expense:,.2f}")
            col3.metric("Credit Card Payments", f"${total_cc_payments:,.2f}")
            col4.metric("Transfers", f"${total_transfers:,.2f}")
            col5.metric("Net Savings", f"${net_savings:,.2f}")

            # ---- ACCOUNT BALANCES ----
            if hasattr(st.session_state, 'account_mappings') and st.session_state.account_mappings:
                st.subheader("💰 Account Balance Changes")
                account_balances = calculate_account_balances(df_year, st.session_state.account_mappings)

                if account_balances:
                    balance_df = pd.DataFrame([
                        {"Account": account, "Net Change": f"${balance:,.2f}",
                         "Direction": "↗️" if balance > 0 else "↘️"}
                        for account, balance in account_balances.items()
                    ])
                    st.dataframe(balance_df, hide_index=True)

            # ---- Category Totals ----
            st.subheader("📂 Category Breakdown")
            if not category_totals.empty:
                st.dataframe(category_totals.reset_index().rename(columns={'amount': 'Total ($)'}))
                if not st.session_state.comp.empty:
                    comp = st.session_state.comp
                    comp['Status'] = comp['Diff'].apply(status_text)

                    # Compact display formatting (no extra padding)
                    # Create a display copy with nicely formatted currency strings for readability
                    display = comp.copy()
                    display['Budget'] = display['Budget'].map(lambda v: f"${v:,.2f}")
                    display['Actual'] = display['Actual'].map(lambda v: f"${v:,.2f}")
                    display['Diff'] = display['Diff'].map(lambda v: f"${v:,.2f}")

                    st.write("### 📊 Budget vs Actual (compact)")
                    st.table(display)  # compact presentation

                    # Quick metrics
                    total_budget = comp['Budget'].sum()
                    total_actual = comp['Actual'].sum()
                    leftover = total_budget - total_actual

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Budget", f"${total_budget:,.2f}")
                    col2.metric("Total Actual", f"${total_actual:,.2f}")
                    col3.metric("Remaining", f"${leftover:,.2f}")
            else:
                st.write("No category data available.")

            # ---- Recurring Charges ----
            st.subheader("🔁 Recurring Charges (Services)")
            recurring_df = find_recurring_charges(df_year)
            if not recurring_df.empty:
                st.dataframe(recurring_df)
            else:
                st.write("No recurring charges detected.")

            # ---- Monthly Trends Table ----
            st.subheader("📆 Monthly Spending Trends (Table)")
            fig, monthly = plot_monthly_trends(df_year)
            if not monthly.empty:
                st.dataframe(monthly)
            else:
                st.write("No monthly data available.")

            # ---- Monthly Transaction ----
            st.subheader("Monthly Transactions")
            # 1. Get a list of unique month names from your DataFrame
            available_months = df_year['month'].unique()

            # 2. Create a chronological month order to sort by
            # This converts month names to datetime objects to determine their correct order
            month_order = pd.to_datetime(available_months, format='%B').sort_values().month_name().unique()

            # 3. Create the selectbox using the sorted list of months
            selected_month = st.selectbox("Select a month", month_order)
            if selected_month:
                month_df = df_year[df_year['month'] == selected_month]
                st.dataframe(month_df)
                expenses_only = month_df[month_df['transaction_type'] == 'Expense']
                category_totals = expenses_only.groupby('category')['amount'].sum()
                st.header("📈 Monthly Visualizations")
                if st.button("Show Monthly Charts", type="primary"):
                    if not category_totals.empty:
                        # Pie chart
                        st.subheader("Expenses by Category (Pie Chart)")
                        fig_pie = plot_pie_chart(category_totals)
                        st.pyplot(fig_pie)

                        # Absolute bar chart
                        st.subheader("Expenses by Category (Absolute $)")
                        fig_bar = plot_bar_chart(category_totals)
                        st.pyplot(fig_bar)

                        # Normalized bar chart
                        st.subheader("Expenses by Category (Normalized %)")
                        fig_norm = plot_normalized_bar(category_totals)
                        st.pyplot(fig_norm)

                        income_only = month_df[month_df['transaction_type'] == 'Income']
                        income_totals = income_only.groupby('category')['amount'].sum()

                        st.subheader("💰 Income by Category")
                        if not income_totals.empty:
                            fig = plot_pie_chart(income_totals)
                            st.pyplot(fig)

                            fig = plot_bar_chart(income_totals)
                            st.pyplot(fig)
                        else:
                            st.info("No income found.")
                    else:
                        st.write("No expense data available for visualization.")

                # ---- CHARTS SECTION ----
            st.header("📈 Yearly Visualizations")
            expenses_only = df[df['transaction_type'] == 'Expense']
            category_totals = expenses_only.groupby('category')['amount'].sum()
            if st.button("Show Yearly Charts", type="primary"):
                if not category_totals.empty:
                    # Monthly trends chart
                    st.subheader("📆 Monthly Spending Trends (Chart)")
                    st.pyplot(fig)

                    # Pie chart
                    st.subheader("Expenses by Category (Pie Chart)")
                    fig_pie = plot_pie_chart(category_totals)
                    st.pyplot(fig_pie)

                    # Absolute bar chart
                    st.subheader("Expenses by Category (Absolute $)")
                    fig_bar = plot_bar_chart(category_totals)
                    st.pyplot(fig_bar)

                    # Normalized bar chart
                    st.subheader("Expenses by Category (Normalized %)")
                    fig_norm = plot_normalized_bar(category_totals)
                    st.pyplot(fig_norm)

                    income_only = df[df['transaction_type'] == 'Income']
                    income_totals = income_only.groupby('category')['amount'].sum()

                    st.subheader("💰 Income by Category")
                    if not income_totals.empty:
                        fig = plot_pie_chart(income_totals)
                        st.pyplot(fig)

                        fig = plot_bar_chart(income_totals)
                        st.pyplot(fig)
                    else:
                        st.info("No income found.")
                else:
                    st.write("No expense data available for visualization.")

            st.subheader("📥 Download Data")

            excel_file = export_to_excel(st.session_state.processed_data)
            st.download_button(
                label="Download as Excel",
                data=excel_file,
                file_name="transactions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            pdf_file = export_to_pdf(
                st.session_state.processed_data,
            )
            st.download_button("Download PDF", data=pdf_file, file_name="transactions.pdf", mime="application/pdf")

    else:
        st.warning("No valid year data found in the processed data.")
