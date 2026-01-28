# =============================================================================
# DATABASE LAYER - SQLite Operations
# =============================================================================
"""
This module handles all database operations for the loan prediction system.
It provides functions to initialize the database, save predictions, and retrieve data.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


class LoanDatabase:
    """Database layer for loan prediction system"""

    def __init__(self, db_name: str = 'loan_predictions.db'):
        """Initialize database connection"""
        self.db_name = db_name
        self.init_database()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_name)

    def init_database(self):
        """Initialize SQLite database and create predictions table"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                married TEXT,
                dependents TEXT,
                education TEXT,
                self_employed TEXT,
                property_area TEXT,
                applicant_income REAL,
                coapplicant_income REAL,
                loan_amount REAL,
                loan_term INTEGER,
                credit_history INTEGER,
                prediction TEXT,
                confidence REAL
            )
        ''')

        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")

    def save_prediction(
        self,
        married: str,
        dependents: str,
        education: str,
        self_employed: str,
        property_area: str,
        applicant_income: float,
        coapplicant_income: float,
        loan_amount: float,
        loan_term: int,
        credit_history: int,
        prediction: str,
        confidence: float
    ) -> int:
        """
        Save a new prediction to the database

        Returns:
            The ID of the inserted record
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO predictions (
                timestamp, married, dependents, education, self_employed,
                property_area, applicant_income, coapplicant_income,
                loan_amount, loan_term, credit_history, prediction, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            married,
            dependents,
            education,
            self_employed,
            property_area,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            prediction,
            confidence
        ))

        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        return record_id

    def get_all_predictions(self) -> List[Dict]:
        """
        Retrieve all predictions from the database

        Returns:
            List of prediction records as dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM predictions ORDER BY timestamp DESC
        ''')

        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        conn.close()

        # Convert to list of dictionaries
        predictions = []
        for row in rows:
            predictions.append(dict(zip(columns, row)))

        return predictions

    def get_predictions_dataframe(self) -> pd.DataFrame:
        """
        Retrieve all predictions as a pandas DataFrame

        Returns:
            DataFrame containing all predictions
        """
        conn = self.get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()
        return df

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """
        Retrieve a specific prediction by ID

        Args:
            prediction_id: The ID of the prediction to retrieve

        Returns:
            Dictionary containing the prediction data, or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM predictions WHERE id = ?
        ''', (prediction_id,))

        columns = [description[0] for description in cursor.description]
        row = cursor.fetchone()

        conn.close()

        if row:
            return dict(zip(columns, row))
        return None

    def get_statistics(self) -> Dict:
        """
        Get statistics about predictions

        Returns:
            Dictionary containing prediction statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        # Approved predictions
        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 'Approved'")
        total_approved = cursor.fetchone()[0]

        # Rejected predictions
        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 'Rejected'")
        total_rejected = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_confidence = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            'total_predictions': total_predictions,
            'total_approved': total_approved,
            'total_rejected': total_rejected,
            'approval_rate': (total_approved / total_predictions * 100) if total_predictions > 0 else 0.0,
            'average_confidence': avg_confidence
        }

    def delete_prediction(self, prediction_id: int) -> bool:
        """
        Delete a prediction by ID

        Args:
            prediction_id: The ID of the prediction to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM predictions WHERE id = ?",
                       (prediction_id,))
        conn.commit()

        deleted = cursor.rowcount > 0
        conn.close()

        return deleted


# Convenience functions for backward compatibility
def init_database():
    """Initialize the database"""
    db = LoanDatabase()
    return db


def save_prediction(input_data, prediction, confidence):
    """Save prediction to database (backward compatible)"""
    db = LoanDatabase()
    return db.save_prediction(
        married=input_data['Married'].iloc[0],
        dependents=input_data['Dependents'].iloc[0],
        education=input_data['Education'].iloc[0],
        self_employed=input_data['Self_Employed'].iloc[0],
        property_area=input_data['Property_Area'].iloc[0],
        applicant_income=input_data['ApplicantIncome'].iloc[0],
        coapplicant_income=input_data['CoapplicantIncome'].iloc[0],
        loan_amount=input_data['LoanAmount'].iloc[0],
        loan_term=input_data['Loan_Amount_Term'].iloc[0],
        credit_history=input_data['Credit_History'].iloc[0],
        prediction='Approved' if prediction == 1 else 'Rejected',
        confidence=confidence
    )


# Test the database layer
if __name__ == "__main__":
    print("Testing Database Layer...")

    db = LoanDatabase()

    # Test saving a prediction
    record_id = db.save_prediction(
        married='Yes',
        dependents='2',
        education='Graduate',
        self_employed='No',
        property_area='Urban',
        applicant_income=5000,
        coapplicant_income=2000,
        loan_amount=150,
        loan_term=360,
        credit_history=1,
        prediction='Approved',
        confidence=85.5
    )
    print(f"✅ Saved prediction with ID: {record_id}")

    # Test retrieving statistics
    stats = db.get_statistics()
    print(f"📊 Statistics: {stats}")

    # Test retrieving all predictions
    predictions = db.get_all_predictions()
    print(f"📋 Total predictions in database: {len(predictions)}")
