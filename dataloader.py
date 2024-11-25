from torch.utils.data import dataloader, random_split
from torchvision import datasets, transforms
from typing import Any
from tqdm import tqdm
import pandas as pd
import requests
import zipfile
import os


def extract_zip_csv(zip_path:str="./data/GalaxyZoo1_DR_table5.csv.zip", extract_to_path:str="./data") -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    
    print("All files extracted")

def assign_galaxy_type(row):
    type_probs = {
        "Elliptical": row["Prob_Elliptical_R1"],
        "ClockwiseSpiral": row["Prob_Spiral_CW_R1"],
        "AntiClockwiseSpiral": row["Prob_Spiral_ACW_R1"],
        "EdgeOn": row["Prob_EdgeOn_R1"],
        "Merger": row["Prob_Merger_R1"],
        "StarArtifact": row["Prob_Star_Artifact_R1"],
    }
    return max(type_probs, key=type_probs.get)

def download(galaxy_type, galaxy_id, ra, dec):
    OUTPUT_DIR = "./data"
    MAIN_ENDPOINT = "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"

    folder_path = os.path.join(OUTPUT_DIR, galaxy_type)
    os.makedirs(folder_path, exist_ok=True)

    imagepath = os.path.join(folder_path, f"{galaxy_id}.jpg")
    if os.path.exists(imagepath) == True:
        pass

    params = {
        "ra": ra,
        "dec": dec,
        "scale": 0.2,      # Adjust as needed
        "width": 256,      # Image width
        "height": 256      # Image height
    }
    response = requests.get(MAIN_ENDPOINT, params=params)

    if response.status_code == 200:
        with open(imagepath, "wb") as file:
            file.write(response.content)

    else:
        print(f"Failed to download for GalaxyID {galaxy_id} (RA: {ra}, DEC: {dec})")

def download_images(df):
    OUTPUT_DIR = "./data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading and Organizing Images"):
        ra = row["RightAscension"]
        dec = row["Declination"]
        galaxy_id = row["GalaxyID"]
        galaxy_type = assign_galaxy_type(row)

        os.makedirs("./data/Elliptical", exist_ok=True)
        os.makedirs("./data/ClockwiseSpiral", exist_ok=True)
        os.makedirs("./data/AntiClockwiseSpiral", exist_ok=True)
        os.makedirs("./data/EdgeOn", exist_ok=True)
        os.makedirs("./data/Merger", exist_ok=True)
        os.makedirs("./data/StarArtifact", exist_ok=True)

        num_file_elliptical = len(os.listdir('./data/Elliptical'))
        num_file_clockwisespiral = len(os.listdir('./data/ClockwiseSpiral'))
        num_file_antimclockwisespiral = len(os.listdir('./data/AntiClockwiseSpiral'))
        num_file_edgeon = len(os.listdir('./data/EdgeOn'))
        num_file_merger = len(os.listdir('./data/Merger'))
        num_file_starartifact = len(os.listdir('./data/StarArtifact'))
        # Extract galaxy metadata
        MAX_IMAGES = 2500
        if galaxy_type == "Elliptical" and num_file_elliptical < MAX_IMAGES:
            download("Elliptical", galaxy_id, ra, dec)
        elif galaxy_type == "ClockwiseSpiral" and num_file_clockwisespiral < MAX_IMAGES:
            download("ClockwiseSpiral", galaxy_id, ra, dec)
        elif galaxy_type == "AntiClockwiseSpiral" and num_file_antimclockwisespiral < MAX_IMAGES:
            download("AntiClockwiseSpiral", galaxy_id, ra, dec)
        elif galaxy_type == "EdgeOn" and num_file_edgeon < MAX_IMAGES:
            download("EdgeOn", galaxy_id, ra, dec)
        elif galaxy_type == "Merger" and num_file_merger < MAX_IMAGES:
            download("Merger", galaxy_id, ra, dec)
        elif galaxy_type == "StarArtifact" and num_file_starartifact < MAX_IMAGES:
            download("StarArtifact", galaxy_id, ra, dec)
        else:
            continue

def main(csv_path:str="./data/GalaxyZoo1_DR_table5.csv"):
    df: pd.DataFrame = pd.read_csv(csv_path)

    df.rename(columns={
        'OBJID': 'GalaxyID',
        'RA': 'RightAscension',
        'DEC': 'Declination',
        'NVOTE_MR1': 'Votes_Round1',
        'P_EL_MR1': 'Prob_Elliptical_R1',
        'P_CW_MR1': 'Prob_Spiral_CW_R1',
        'P_ACW_MR1': 'Prob_Spiral_ACW_R1',
        'P_EDGE_MR1': 'Prob_EdgeOn_R1',
        'P_DK_MR1': 'Prob_Unknown_R1',
        'P_MG_MR1': 'Prob_Merger_R1',
        'P_CS_MR1': 'Prob_Star_Artifact_R1',
        'NVOTE_MR2': 'Votes_Round2',
        'P_EL_MR2': 'Prob_Elliptical_R2',
        'P_CW_MR2': 'Prob_Spiral_CW_R2',
        'P_ACW_MR2': 'Prob_Spiral_ACW_R2',
        'P_EDGE_MR2': 'Prob_EdgeOn_R2',
        'P_DK_MR2': 'Prob_Unknown_R2',
        'P_MG_MR2': 'Prob_Merger_R2',
        'P_CS_MR2': 'Prob_Star_Artifact_R2',
    }, inplace=True)

    download_images(df)

    columns_to_drop = ['RightAscension', 'Declination', 'Votes_Round1', 'Votes_Round2',
                       'Prob_Unknown_R1', 'Prob_Unknown_R2']
    df.drop(columns=columns_to_drop, inplace=True)
    
    df['IsElliptical1'] = (df['Prob_Elliptical_R1'] > 0.5).astype(int)
    df['IsElliptical2'] = (df['Prob_Elliptical_R2'] > 0.5).astype(int)
    df['IsSpiral1'] = (df['Prob_Spiral_CW_R1'] > 0.5).astype(int)
    df['IsSpiral2'] = (df["Prob_Spiral_CW_R2"] > 0.5).astype(int)
    df['IsAntiClockwiseSpiral1'] = (df['Prob_Spiral_ACW_R1'] > 0.5).astype(int)
    df['IsAntiClockwiseSpiral2'] = (df['Prob_Spiral_ACW_R2'] > 0.5).astype(int)
    df['IsEdgeOn1'] = (df['Prob_EdgeOn_R1'] > 0.5).astype(int)
    df['IsEdgeOn2'] = (df['Prob_EdgeOn_R2'] > 0.5).astype(int)
    df['IsMergin1'] = (df['Prob_Merger_R1'] > 0.5).astype(int)
    df['IsMergin2'] = (df['Prob_Merger_R2'] > 0.5).astype(int)
    df['IsStarOrArtifact1'] = (df["Prob_Star_Artifact_R1"] > 0.5).astype(int)
    df['IsStarOrArtifact2'] = (df["Prob_Star_Artifact_R2"] > 0.5).astype(int)

    columns_to_drop = ["Prob_Elliptical_R1", "Prob_Elliptical_R2", "Prob_Spiral_CW_R1", "Prob_Spiral_CW_R2",
                       "Prob_Spiral_ACW_R1", "Prob_Spiral_ACW_R2", "Prob_EdgeOn_R1", "Prob_EdgeOn_R2",
                       "Prob_Merger_R1", "Prob_Merger_R2", "Prob_Star_Artifact_R1", "Prob_Star_Artifact_R2"]

    df.drop(columns=columns_to_drop, inplace=True)
    df.fillna(0, inplace=True)


def get_dataloader(BATCHSIZE:int=34):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root="./data", transform=transform)
    print(dataset)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = dataloader.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

    return train_loader, test_loader, dataset.class_to_idx


if __name__ == "__main__":
    # extract_zip_csv()
    # main()
    get_dataloader()
