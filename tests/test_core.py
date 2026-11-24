from docrouter_core import FOLDERS, REVIEW_FOLDER, DocumentType, destination_for


def test_every_type_has_a_folder():
    assert set(FOLDERS) == set(DocumentType)


def test_confident_prediction_goes_to_type_folder():
    assert destination_for(DocumentType.FACTURE, 0.9, 0.6) == "factures"


def test_uncertain_prediction_goes_to_review():
    assert destination_for(DocumentType.FACTURE, 0.59, 0.6) == REVIEW_FOLDER


def test_threshold_is_inclusive():
    assert destination_for(DocumentType.AUTRE, 0.6, 0.6) == "autres"
