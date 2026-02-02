"""
Database layer for storing house sales predictions
"""
import sqlite3
from datetime import datetime
from typing import Dict, List
import json


class PredictionDatabase:
    def __init__(self, db_path='house_predictions.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with tables for predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table for price predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                square_footage REAL,
                bedrooms INTEGER,
                bathrooms REAL,
                age INTEGER,
                garage_spaces REAL,
                lot_size REAL,
                floors INTEGER,
                neighborhood_rating INTEGER,
                condition INTEGER,
                school_rating REAL,
                has_pool INTEGER,
                renovated INTEGER,
                location_type TEXT,
                distance_to_center_km REAL,
                days_on_market REAL,
                predicted_price REAL NOT NULL
            )
        ''')

        # Create table for sold predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sold_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                square_footage REAL,
                bedrooms INTEGER,
                bathrooms REAL,
                age INTEGER,
                garage_spaces REAL,
                lot_size REAL,
                floors INTEGER,
                neighborhood_rating INTEGER,
                condition INTEGER,
                school_rating REAL,
                has_pool INTEGER,
                renovated INTEGER,
                location_type TEXT,
                distance_to_center_km REAL,
                days_on_market REAL,
                predicted_sold INTEGER NOT NULL,
                probability REAL
            )
        ''')

        conn.commit()
        conn.close()
        print("Database initialized successfully")

    def save_price_prediction(self, features: Dict, predicted_price: float):
        """Save a price prediction to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO price_predictions (
                timestamp, square_footage, bedrooms, bathrooms, age, garage_spaces,
                lot_size, floors, neighborhood_rating, condition, school_rating,
                has_pool, renovated, location_type, distance_to_center_km,
                days_on_market, predicted_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            features.get('Square_Footage'),
            features.get('Bedrooms'),
            features.get('Bathrooms'),
            features.get('Age'),
            features.get('Garage_Spaces'),
            features.get('Lot_Size'),
            features.get('Floors'),
            features.get('Neighborhood_Rating'),
            features.get('Condition'),
            features.get('School_Rating'),
            features.get('Has_Pool'),
            features.get('Renovated'),
            features.get('Location_Type'),
            features.get('Distance_To_Center_KM'),
            features.get('Days_On_Market'),
            predicted_price
        ))

        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        return prediction_id

    def save_sold_prediction(self, features: Dict, predicted_sold: int, probability: float):
        """Save a sold prediction to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO sold_predictions (
                timestamp, square_footage, bedrooms, bathrooms, age, garage_spaces,
                lot_size, floors, neighborhood_rating, condition, school_rating,
                has_pool, renovated, location_type, distance_to_center_km,
                days_on_market, predicted_sold, probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            features.get('Square_Footage'),
            features.get('Bedrooms'),
            features.get('Bathrooms'),
            features.get('Age'),
            features.get('Garage_Spaces'),
            features.get('Lot_Size'),
            features.get('Floors'),
            features.get('Neighborhood_Rating'),
            features.get('Condition'),
            features.get('School_Rating'),
            features.get('Has_Pool'),
            features.get('Renovated'),
            features.get('Location_Type'),
            features.get('Distance_To_Center_KM'),
            features.get('Days_On_Market'),
            predicted_sold,
            probability
        ))

        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        return prediction_id

    def get_recent_predictions(self, table_name: str, limit: int = 10) -> List[Dict]:
        """Get recent predictions from specified table"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(f'''
            SELECT * FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_statistics(self) -> Dict:
        """Get statistics about predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Price predictions stats
        cursor.execute(
            'SELECT COUNT(*), AVG(predicted_price), MIN(predicted_price), MAX(predicted_price) FROM price_predictions')
        price_stats = cursor.fetchone()

        # Sold predictions stats
        cursor.execute(
            'SELECT COUNT(*), SUM(predicted_sold) FROM sold_predictions')
        sold_stats = cursor.fetchone()

        conn.close()

        return {
            'price_predictions': {
                'total': price_stats[0] or 0,
                'avg_price': price_stats[1] or 0,
                'min_price': price_stats[2] or 0,
                'max_price': price_stats[3] or 0
            },
            'sold_predictions': {
                'total': sold_stats[0] or 0,
                'predicted_sold': sold_stats[1] or 0
            }
        }


if __name__ == '__main__':
    # Test the database
    db = PredictionDatabase()
    print("Database setup complete!")
    print(db.get_statistics())
