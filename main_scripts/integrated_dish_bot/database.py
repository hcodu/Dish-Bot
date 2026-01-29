"""
Database operations for Dish Bot
SQLite database for storing dish detection events
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


class Database:
    """SQLite database handler for dish events"""

    def __init__(self, db_path='dish_bot.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn

    def init_database(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create dish_events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dish_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dish_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                date DATE NOT NULL,
                person_name TEXT NOT NULL,
                confidence_score REAL,
                dish_snapshot_path TEXT,
                face_snapshot_path TEXT,
                dish_bbox TEXT,
                face_bbox TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON dish_events(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person ON dish_events(person_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON dish_events(timestamp)')

        conn.commit()
        conn.close()
        print("✓ Database initialized")

    def insert_dish_event(
        self,
        dish_id: int,
        timestamp: datetime,
        person_name: str,
        confidence: float,
        dish_snapshot_path: str,
        face_snapshot_path: Optional[str],
        dish_bbox: str,
        face_bbox: Optional[str]
    ) -> int:
        """
        Insert a new dish detection event

        Args:
            dish_id: Dish tracking ID
            timestamp: When dish was first detected
            person_name: Identified person name (or "Unknown")
            confidence: Face recognition confidence score
            dish_snapshot_path: Path to dish image
            face_snapshot_path: Path to face image (or None)
            dish_bbox: JSON string of dish bounding box
            face_bbox: JSON string of face bounding box (or None)

        Returns:
            ID of inserted event
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        date = timestamp.date()

        cursor.execute('''
            INSERT INTO dish_events (
                dish_id, timestamp, date, person_name, confidence_score,
                dish_snapshot_path, face_snapshot_path, dish_bbox, face_bbox
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dish_id, timestamp, date, person_name, confidence,
            dish_snapshot_path, face_snapshot_path, dish_bbox, face_bbox
        ))

        event_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"✓ Dish event {event_id} saved to database")
        return event_id

    def get_events_by_date(self, date) -> List[Dict]:
        """Get all events for a specific date"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM dish_events
            WHERE date = ?
            ORDER BY timestamp DESC
        ''', (date,))

        events = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return events

    def get_events_paginated(self, page: int = 1, per_page: int = 20) -> Dict:
        """
        Get paginated events for history page

        Args:
            page: Page number (1-indexed)
            per_page: Events per page

        Returns:
            Dictionary with events, total count, and pagination info
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get total count
        cursor.execute('SELECT COUNT(*) as count FROM dish_events')
        total = cursor.fetchone()['count']

        # Get paginated results
        offset = (page - 1) * per_page
        cursor.execute('''
            SELECT * FROM dish_events
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', (per_page, offset))

        events = [dict(row) for row in cursor.fetchall()]
        conn.close()

        total_pages = (total + per_page - 1) // per_page  # Ceiling division

        return {
            'events': events,
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': total_pages,
            'has_prev': page > 1,
            'has_next': page < total_pages
        }

    def get_daily_stats(self, date) -> Dict:
        """
        Get aggregated statistics for a specific day

        Args:
            date: Date to get stats for

        Returns:
            Dictionary with total dishes and breakdown by person
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get count by person
        cursor.execute('''
            SELECT person_name, COUNT(*) as count
            FROM dish_events
            WHERE date = ?
            GROUP BY person_name
            ORDER BY count DESC
        ''', (date,))

        by_person = {row['person_name']: row['count'] for row in cursor.fetchall()}
        total_dishes = sum(by_person.values())

        conn.close()

        return {
            'date': str(date),
            'total_dishes': total_dishes,
            'by_person': by_person
        }

    def get_events_before_date(self, cutoff_date: datetime) -> List[Dict]:
        """
        Get all events before a certain date (for cleanup)

        Args:
            cutoff_date: Get events older than this date

        Returns:
            List of event dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM dish_events
            WHERE timestamp < ?
            AND (dish_snapshot_path IS NOT NULL OR face_snapshot_path IS NOT NULL)
        ''', (cutoff_date,))

        events = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return events

    def clear_snapshot_paths(self, event_id: int):
        """
        Set snapshot paths to NULL for an event (after files are deleted)

        Args:
            event_id: ID of event to update
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE dish_events
            SET dish_snapshot_path = NULL,
                face_snapshot_path = NULL
            WHERE id = ?
        ''', (event_id,))

        conn.commit()
        conn.close()

    def get_all_events(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all events (optionally limited)

        Args:
            limit: Maximum number of events to return (None for all)

        Returns:
            List of event dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if limit:
            cursor.execute('''
                SELECT * FROM dish_events
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT * FROM dish_events
                ORDER BY timestamp DESC
            ''')

        events = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return events


if __name__ == '__main__':
    # Test database
    db = Database('test_dish_bot.db')

    # Insert test event
    test_timestamp = datetime.now()
    event_id = db.insert_dish_event(
        dish_id=1,
        timestamp=test_timestamp,
        person_name="John",
        confidence=45.2,
        dish_snapshot_path="static/snapshots/dish_1_test.jpg",
        face_snapshot_path="static/snapshots/face_1_test.jpg",
        dish_bbox=json.dumps({"x1": 100, "y1": 200, "x2": 300, "y2": 400}),
        face_bbox=json.dumps({"x": 50, "y": 60, "w": 100, "h": 120})
    )

    print(f"\nTest event ID: {event_id}")

    # Get today's events
    today = datetime.now().date()
    events = db.get_events_by_date(today)
    print(f"\nEvents today: {len(events)}")

    # Get stats
    stats = db.get_daily_stats(today)
    print(f"\nDaily stats: {stats}")

    # Cleanup test db
    os.remove('test_dish_bot.db')
    print("\n✓ Test passed, database cleaned up")
