"""
Optional Centralized SQLite Storage Layer
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Schema:
- FILES: Stores indexed file metadata and SHA-256 signatures.
- DETECTION_RESULTS: Stores detected duplicate relations and metrics.
- SCAN_HISTORY: Stores audit logs of scans conducted.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from result_manager import DuplicateRecord

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "duplicate_detection.db"


class DatabaseManager:
    """Manages SQLite storage for files, detection results, and scan audits."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        """Initializes tables according to the requested architecture."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. FILES table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    sha256_hash TEXT,
                    file_type TEXT NOT NULL,
                    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 2. DETECTION_RESULTS table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_a_id INTEGER NOT NULL,
                    file_b_id INTEGER NOT NULL,
                    duplicate_type TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    detection_algorithm TEXT NOT NULL,
                    detection_details TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_a_id) REFERENCES files (id),
                    FOREIGN KEY (file_b_id) REFERENCES files (id),
                    UNIQUE (file_a_id, file_b_id, detection_algorithm)
                );
                """
            )

            # 3. CLASSIFICATION_RESULTS table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS classification_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                );
                """
            )

            # 4. SCAN_HISTORY table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scanned_folder TEXT NOT NULL,
                    total_files INTEGER NOT NULL,
                    total_matches INTEGER NOT NULL,
                    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scan_status TEXT NOT NULL
                );
                """
            )

            conn.commit()

    def upsert_file(
        self,
        file_path: str,
        filename: str,
        extension: str,
        file_size: int,
        file_type: str,
        sha256_hash: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """Inserts or updates a file record and returns its database ID."""
        should_close = False
        if conn is None:
            conn = self._get_connection()
            should_close = True

        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO files (file_path, filename, extension, file_size, sha256_hash, file_type)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                filename = excluded.filename,
                file_size = excluded.file_size,
                sha256_hash = COALESCE(excluded.sha256_hash, files.sha256_hash),
                file_type = excluded.file_type;
            """,
            (file_path, filename, extension, file_size, sha256_hash, file_type),
        )
        cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        conn.commit()

        if should_close:
            conn.close()

        return row[0]

    def save_results(self, records: List[DuplicateRecord]) -> int:
        """Stores duplicate candidate records linking to file IDs."""
        saved_count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for rec in records:
                # Ensure both files exist in the files table
                id_a = self.upsert_file(
                    file_path=rec.file_path_1,
                    filename=rec.file_name_1,
                    extension=Path(rec.file_path_1).suffix.lower(),
                    file_size=Path(rec.file_path_1).stat().st_size if Path(rec.file_path_1).exists() else 0,
                    file_type=rec.file_type,
                    conn=conn,
                )
                id_b = self.upsert_file(
                    file_path=rec.file_path_2,
                    filename=rec.file_name_2,
                    extension=Path(rec.file_path_2).suffix.lower(),
                    file_size=Path(rec.file_path_2).stat().st_size if Path(rec.file_path_2).exists() else 0,
                    file_type=rec.file_type,
                    conn=conn,
                )

                file_min, file_max = min(id_a, id_b), max(id_a, id_b)

                cursor.execute(
                    """
                    INSERT INTO detection_results (
                        file_a_id, file_b_id, duplicate_type, similarity_score,
                        detection_algorithm, detection_details
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_a_id, file_b_id, detection_algorithm) DO UPDATE SET
                        duplicate_type = excluded.duplicate_type,
                        similarity_score = excluded.similarity_score,
                        detection_details = excluded.detection_details;
                    """,
                    (
                        file_min,
                        file_max,
                        rec.duplicate_type,
                        rec.similarity_score,
                        rec.detection_algorithm,
                        rec.detection_details,
                    ),
                )
                saved_count += 1

            conn.commit()
        return saved_count

    def save_classification_results(self, results: list) -> int:
        """Stores document classification predictions."""
        saved_count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for r in results:
                file_id = self.upsert_file(
                    file_path=r.file_path,
                    filename=r.file_name,
                    extension=Path(r.file_path).suffix.lower(),
                    file_size=Path(r.file_path).stat().st_size if Path(r.file_path).exists() else 0,
                    file_type="document",
                    conn=conn,
                )
                cursor.execute(
                    """
                    INSERT INTO classification_results (file_id, predicted_class, confidence, model_name)
                    VALUES (?, ?, ?, ?);
                    """,
                    (file_id, r.predicted_category, r.confidence, r.model_name),
                )
                saved_count += 1
            conn.commit()
        return saved_count

    def log_scan(
        self,
        scanned_folder: str,
        total_files: int,
        total_matches: int,
        status: str = "COMPLETED",
    ) -> int:
        """Logs execution audit record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scan_history (scanned_folder, total_files, total_matches, scan_status)
                VALUES (?, ?, ?, ?);
                """,
                (scanned_folder, total_files, total_matches, status),
            )
            scan_id = cursor.lastrowid
            conn.commit()
            return scan_id

    def get_database_stats(self) -> Dict[str, int]:
        """Returns aggregate row counts across all managed tables."""
        stats = {}
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for table in ("files", "detection_results", "classification_results", "scan_history"):
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
        finally:
            conn.close()
        return stats

    def get_detection_results(
        self,
        min_similarity: float = 0.0,
        duplicate_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Queries detected duplicate candidate pairs with joined file details."""
        query = """
            SELECT 
                r.id, fa.file_path AS file_path_1, fa.filename AS file_name_1,
                fb.file_path AS file_path_2, fb.filename AS file_name_2,
                r.duplicate_type, r.similarity_score, r.detection_algorithm,
                r.detection_details, r.detected_at
            FROM detection_results r
            JOIN files fa ON r.file_a_id = fa.id
            JOIN files fb ON r.file_b_id = fb.id
            WHERE r.similarity_score >= ?
        """
        params = [min_similarity]
        if duplicate_type and duplicate_type != "All":
            query += " AND r.duplicate_type = ?"
            params.append(duplicate_type)
        query += " ORDER BY r.similarity_score DESC, r.detected_at DESC LIMIT ?"
        params.append(limit)

        results = []
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                results.append(dict(row))
        finally:
            conn.close()
        return results

    def get_scan_history(self, limit: int = 50) -> List[Dict[str, Union[str, int]]]:
        """Queries historical scan execution audit logs."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM scan_history ORDER BY scan_timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_classification_results(self, limit: int = 100) -> List[Dict[str, Union[str, float, int]]]:
        """Queries historical document classification predictions."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.id, f.file_path, f.filename, c.predicted_class, c.confidence, c.model_name, c.classified_at
                FROM classification_results c
                JOIN files f ON c.file_id = f.id
                ORDER BY c.classified_at DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
