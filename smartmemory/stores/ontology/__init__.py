import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict

from smartmemory.ontology.models import Ontology


class OntologyStorage(ABC):
    """Abstract interface for ontology storage backends."""

    @abstractmethod
    def save_ontology(self, ontology: Ontology) -> None:
        """Save an ontology."""
        pass

    @abstractmethod
    def load_ontology(self, ontology_id: str) -> Optional[Ontology]:
        """Load an ontology by ID."""
        pass

    @abstractmethod
    def list_ontologies(self) -> List[Dict[str, str]]:
        """List all available ontologies with basic metadata."""
        pass

    @abstractmethod
    def delete_ontology(self, ontology_id: str) -> bool:
        """Delete an ontology."""
        pass


class FileSystemOntologyStorage(OntologyStorage):
    """File system-based ontology storage."""

    def __init__(self, storage_dir: str = "ontologies"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def save_ontology(self, ontology: Ontology) -> None:
        """Save ontology to JSON file."""
        file_path = self.storage_dir / f"{ontology.id}.json"
        with open(file_path, 'w') as f:
            json.dump(ontology.to_dict(), f, indent=2, default=str)

    def load_ontology(self, ontology_id: str) -> Optional[Ontology]:
        """Load ontology from JSON file."""
        file_path = self.storage_dir / f"{ontology_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            data = json.load(f)

        return Ontology.from_dict(data)

    def list_ontologies(self) -> List[Dict[str, str]]:
        """List all ontology files."""
        ontologies = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                ontologies.append({
                    'id': data['id'],
                    'name': data['name'],
                    'version': data['version'],
                    'domain': data.get('domain', ''),
                    'created_at': data['created_at']
                })
            except Exception:
                continue  # Skip corrupted files
        return ontologies

    def delete_ontology(self, ontology_id: str) -> bool:
        """Delete ontology file."""
        file_path = self.storage_dir / f"{ontology_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
