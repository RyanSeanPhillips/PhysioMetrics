"""
Animal ViewModel — provides animal-centric view of experiment metadata.

MVVM: wraps ProjectService queries, exposes signals for UI binding.
No direct widget manipulation.
"""

from typing import List, Dict, Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.services.project_service import ProjectService


class AnimalViewModel(QObject):
    """ViewModel for the animal-centric Source Review panel."""

    animals_changed = pyqtSignal()
    animal_selected = pyqtSignal(str)  # animal_id

    def __init__(self, service: ProjectService, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._service = service

    def get_animal_summaries(self) -> List[Dict[str, Any]]:
        """Get summary info for each animal.

        Returns list of:
            {animal_id, experiment_count, source_count, has_disagreements, experiments: [...]}
        """
        experiments, _ = self._service.store.get_experiments()

        # Group by animal_id
        animal_map: Dict[str, List[Dict]] = {}
        for exp in experiments:
            aid = exp.get("animal_id", "")
            if aid:
                animal_map.setdefault(aid, []).append(exp)

        # Build summaries
        summaries = []
        for animal_id in sorted(animal_map.keys()):
            exps = animal_map[animal_id]
            links = self._service.get_source_links(animal_id=animal_id)
            disagreements = self._service.detect_disagreements(animal_id=animal_id)

            summaries.append({
                "animal_id": animal_id,
                "experiment_count": len(exps),
                "source_count": len(set(l.get("source_id") for l in links)),
                "link_count": len(links),
                "has_disagreements": len(disagreements) > 0,
                "disagreement_count": len(disagreements),
            })

        return summaries

    def get_animal_detail(self, animal_id: str) -> Dict[str, Any]:
        """Get full detail for an animal: experiments, source_links, disagreements."""
        experiments = self._service.get_experiments_for_animal(animal_id)
        links = self._service.get_source_links(animal_id=animal_id)
        disagreements = self._service.detect_disagreements(animal_id=animal_id)

        # Group links by field
        links_by_field: Dict[str, List[Dict]] = {}
        for link in links:
            field = link.get("field", "unknown")
            links_by_field.setdefault(field, []).append(link)

        return {
            "animal_id": animal_id,
            "experiments": experiments,
            "source_links": links,
            "links_by_field": links_by_field,
            "disagreements": disagreements,
        }

    def get_all_animals(self) -> List[str]:
        """Get sorted list of all unique animal IDs."""
        return self._service.get_unique_values("animal_id")
