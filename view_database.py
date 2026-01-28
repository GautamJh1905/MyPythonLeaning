"""
Quick script to access and view the loan predictions database
"""
from db_layer import LoanDatabase
import pandas as pd

# Initialize database connection
db = LoanDatabase()

print("=" * 60)
print("LOAN PREDICTIONS DATABASE")
print("=" * 60)

# Get statistics
stats = db.get_statistics()
print("\n📊 STATISTICS:")
print(f"  Total Predictions: {stats['total_predictions']}")
print(f"  Approved: {stats['total_approved']}")
print(f"  Rejected: {stats['total_rejected']}")
print(f"  Approval Rate: {stats['approval_rate']:.1f}%")
print(f"  Average Confidence: {stats['average_confidence']:.2f}%")

# Get all predictions
print("\n📋 ALL PREDICTIONS:")
predictions = db.get_all_predictions()
if predictions:
    # Convert to DataFrame for better display
    df = pd.DataFrame(predictions)
    print(df.to_string())

    # Save to CSV
    df.to_csv('loan_predictions_export.csv', index=False)
    print(
        f"\n✅ Exported {len(predictions)} records to 'loan_predictions_export.csv'")
else:
    print("  No predictions found in database")

print("\n" + "=" * 60)
