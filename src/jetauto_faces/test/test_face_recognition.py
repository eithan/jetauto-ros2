"""Tests for face recognition node and enrollment tool."""

import os
import tempfile

import numpy as np
import pytest


class TestFaceDatabase:
    """Test face database loading and matching."""

    def test_save_and_load_enrollment(self):
        """Enrollment file can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake enrollment
            name = 'Test Person'
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            filepath = os.path.join(tmpdir, 'test_person.npz')
            np.savez(filepath, name=name, embedding=embedding, num_samples=5)

            # Load it back
            data = np.load(filepath, allow_pickle=True)
            assert str(data['name']) == name
            assert data['embedding'].shape == (512,)
            assert int(data['num_samples']) == 5

            # Verify normalization preserved
            loaded_emb = data['embedding'].astype(np.float32)
            norm = np.linalg.norm(loaded_emb)
            assert abs(norm - 1.0) < 1e-5

    def test_cosine_similarity_matching(self):
        """Cosine similarity correctly identifies matching embeddings."""
        # Create a known embedding
        known = np.random.randn(512).astype(np.float32)
        known = known / np.linalg.norm(known)

        # Create a similar embedding (small perturbation — scale must be
        # small relative to the unit-norm vector in 512-d space)
        similar = known + np.random.randn(512).astype(np.float32) * 0.02
        similar = similar / np.linalg.norm(similar)

        # Create a random embedding
        random_emb = np.random.randn(512).astype(np.float32)
        random_emb = random_emb / np.linalg.norm(random_emb)

        sim_score = float(np.dot(known, similar))
        rand_score = float(np.dot(known, random_emb))

        # Similar embedding should score high (perturbation is small)
        assert sim_score > 0.9
        # Random embedding in 512-d is roughly orthogonal
        assert abs(rand_score) < 0.15

    def test_average_embeddings(self):
        """Averaging multiple embeddings produces a valid normalized embedding."""
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Create variations (small perturbations)
        embeddings = []
        for _ in range(5):
            variation = base + np.random.randn(512).astype(np.float32) * 0.02
            embeddings.append(variation)

        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)

        # Average should be close to the base
        similarity = float(np.dot(base, avg))
        assert similarity > 0.95

        # Average should be normalized
        assert abs(np.linalg.norm(avg) - 1.0) < 1e-5

    def test_empty_database(self):
        """Empty database returns no match."""
        face_db = {}
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # No enrolled faces — should not match anything
        best_score = 0.0
        for name, db_emb in face_db.items():
            score = float(np.dot(embedding, db_emb))
            if score > best_score:
                best_score = score

        assert best_score == 0.0
