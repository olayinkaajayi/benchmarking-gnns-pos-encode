"""
    File to load dataset based on user control from main file
"""
from data.COLLAB import COLLABDataset
from data.WikiCS import WikiCSDataset
from data.Pubmed import PubmedDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    # handling for COLLAB dataset
    if DATASET_NAME == 'OGBL-COLLAB':
        return COLLABDataset(DATASET_NAME)

    if DATASET_NAME == 'WikiCS':
        return WikiCSDataset(DATASET_NAME)

    if DATASET_NAME == 'Pubmed':
        return PubmedDataset(DATASET_NAME)
