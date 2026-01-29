"""
Media Cleanup Module
Automatically delete snapshot files older than 7 days
"""

import os
import time
import threading
from datetime import datetime, timedelta
from database import Database


class MediaCleaner:
    """Background thread to clean up old media files"""

    def __init__(self, db: Database, retention_days: int = 7):
        """
        Initialize media cleaner

        Args:
            db: Database instance
            retention_days: Number of days to retain media files
        """
        self.db = db
        self.retention_days = retention_days
        self.running = False
        self.thread = None

    def cleanup_old_media(self):
        """Delete snapshot files older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        print(f"\n{'='*60}")
        print(f"MEDIA CLEANUP - Deleting files older than {cutoff_date.date()}")
        print(f"{'='*60}")

        # Get old events
        old_events = self.db.get_events_before_date(cutoff_date)

        if not old_events:
            print("  ℹ️  No old media files to clean up")
            print(f"{'='*60}\n")
            return

        deleted_count = 0
        for event in old_events:
            # Delete dish snapshot
            if event['dish_snapshot_path'] and os.path.exists(event['dish_snapshot_path']):
                try:
                    os.remove(event['dish_snapshot_path'])
                    deleted_count += 1
                except Exception as e:
                    print(f"  ⚠️  Error deleting {event['dish_snapshot_path']}: {e}")

            # Delete face snapshot
            if event['face_snapshot_path'] and os.path.exists(event['face_snapshot_path']):
                try:
                    os.remove(event['face_snapshot_path'])
                    deleted_count += 1
                except Exception as e:
                    print(f"  ⚠️  Error deleting {event['face_snapshot_path']}: {e}")

            # Update database to clear paths
            self.db.clear_snapshot_paths(event['id'])

        print(f"  ✓ Deleted {deleted_count} old snapshot files")
        print(f"  ✓ Updated {len(old_events)} database records")
        print(f"{'='*60}\n")

    def run_cleanup_loop(self):
        """Background loop that runs cleanup daily"""
        print(f"✓ Media cleanup thread started (retention: {self.retention_days} days)")

        while self.running:
            # Run cleanup
            try:
                self.cleanup_old_media()
            except Exception as e:
                print(f"✗ Error in media cleanup: {e}")

            # Sleep for 24 hours
            # Check every minute if we should stop
            for _ in range(24 * 60):
                if not self.running:
                    break
                time.sleep(60)

        print("Media cleanup thread stopped")

    def start(self):
        """Start the cleanup thread"""
        if self.running:
            print("⚠️  Media cleaner already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self.run_cleanup_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the cleanup thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def manual_cleanup(self):
        """Manually trigger a cleanup (for testing)"""
        print("Manual cleanup triggered...")
        self.cleanup_old_media()


if __name__ == '__main__':
    # Test media cleanup
    print("Testing Media Cleanup Module")
    print("=" * 60)

    db = Database('test_dish_bot.db')

    # Create cleaner
    cleaner = MediaCleaner(db, retention_days=7)

    # Run manual cleanup
    cleaner.manual_cleanup()

    # Cleanup test db
    if os.path.exists('test_dish_bot.db'):
        os.remove('test_dish_bot.db')

    print("\n✓ Test passed")
