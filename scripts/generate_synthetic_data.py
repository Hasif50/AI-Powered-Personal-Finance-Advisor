"""
Synthetic Data Generation Script for AI-Powered Personal Finance Advisor
From Hasif's Workspace

Generates realistic financial transaction data for testing and demonstration.
"""

import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import os
import argparse

# Initialize Faker
fake = Faker()

def generate_transactions(num_transactions: int = 1000) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame with synthetic financial transaction data.
    
    Args:
        num_transactions (int): The number of transactions to generate.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the synthetic transaction data.
    """
    categories = [
        'Groceries', 'Utilities', 'Rent/Mortgage', 'Transportation', 
        'Entertainment', 'Healthcare', 'Dining Out', 'Shopping', 
        'Income', 'Other'
    ]
    
    transaction_data = []
    
    for _ in range(num_transactions):
        # Generate random date within the last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2 * 365)
        transaction_date = fake.date_between(start_date=start_date, end_date=end_date)
        
        # Generate realistic transaction description
        description_type = random.choice(['company', 'product', 'service', 'generic_store'])
        if description_type == 'company':
            description = f"{fake.company()} {random.choice(['Services', 'Payment', 'Purchase'])}"
        elif description_type == 'product':
            description = f"Purchase of {fake.word()} {fake.word()}"
        elif description_type == 'service':
            description = f"{fake.catch_phrase()} service"
        else:  # generic_store
            description = f"{random.choice(['Shopping at ', 'Payment to '])}{fake.company_suffix()} {fake.company()}"
        
        # Generate random amount
        amount = round(random.uniform(5.00, 1000.00), 2)
        
        # Choose a random category
        category = random.choice(categories)
        
        # Determine transaction type
        if category == 'Income':
            transaction_type = 'Credit'
        else:
            # For other categories, mostly Debit, but allow some Credits (e.g., refunds)
            transaction_type = random.choices(['Debit', 'Credit'], weights=[0.9, 0.1], k=1)[0]
        
        transaction_data.append({
            'date': transaction_date,
            'description': description,
            'amount': amount,
            'category': category,
            'transaction_type': transaction_type
        })
    
    df = pd.DataFrame(transaction_data)
    return df

def main():
    """Main function to generate and save synthetic data."""
    parser = argparse.ArgumentParser(description='Generate synthetic financial transaction data')
    parser.add_argument('--num_transactions', type=int, default=1000, 
                       help='Number of transactions to generate (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='data', 
                       help='Output directory (default: data)')
    parser.add_argument('--output_file', type=str, default='synthetic_transactions.csv',
                       help='Output filename (default: synthetic_transactions.csv)')
    
    args = parser.parse_args()
    
    NUM_TRANSACTIONS = args.num_transactions
    OUTPUT_DIR = args.output_dir
    OUTPUT_FILE = args.output_file
    
    print(f"Generating {NUM_TRANSACTIONS} synthetic transactions...")
    
    # Generate the transactions
    transactions_df = generate_transactions(NUM_TRANSACTIONS)
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Full output path
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Save the DataFrame to a CSV file
    transactions_df.to_csv(output_path, index=False)
    
    print(f"Successfully generated and saved {NUM_TRANSACTIONS} transactions to {output_path}")
    
    # Display summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Date range: {transactions_df['date'].min()} to {transactions_df['date'].max()}")
    print(f"Total amount: ${transactions_df['amount'].sum():,.2f}")
    print(f"Average transaction: ${transactions_df['amount'].mean():.2f}")
    print(f"Categories: {', '.join(transactions_df['category'].unique())}")
    print(f"Transaction types: {', '.join(transactions_df['transaction_type'].unique())}")
    
    # Category breakdown
    print("\n--- Category Breakdown ---")
    category_summary = transactions_df.groupby('category').agg({
        'amount': ['count', 'sum', 'mean']
    }).round(2)
    category_summary.columns = ['Count', 'Total Amount', 'Avg Amount']
    print(category_summary)

if __name__ == "__main__":
    main()
