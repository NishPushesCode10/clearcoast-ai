import os
import requests
from dotenv import load_dotenv
import planetary_computer
from pystac_client import Client
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# Load environment variables from .env file
load_dotenv()

def download_s2_data():
    """
    Downloads a cloudy Sentinel-2 L2A tile over Chennai Marina Beach + Ennore Creek
    using Microsoft Planetary Computer and STAC.
    """
    print("Connecting to Microsoft Planetary Computer...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Bounding box for Chennai Marina Beach + Ennore Creek
    bbox = [80.25, 13.05, 80.35, 13.15]
    
    print("Searching for Sentinel-2 L2A tiles with >30% cloud cover...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        # Restrict to last couple of years to find a good recent match
        datetime="2023-01-01/2026-12-31", 
        query={"eo:cloud_cover": {"gte": 30}}
    )

    items = list(search.items())
    if not items:
        raise ValueError("No matching cloudy Sentinel-2 tiles found in Planetary Computer.")

    # Sort to pick the most recent one
    items.sort(key=lambda x: x.datetime, reverse=True)
    item = items[0]
    
    date_str = item.datetime.strftime("%Y-%m-%d")
    output_filename = f"chennai_s2_cloudy_{date_str}.tif"
    
    print(f"Found tile from {date_str} with {item.properties['eo:cloud_cover']}% cloud cover.")
    
    # The 'visual' asset is a 3-band true color GeoTIFF (rendered RGB)
    asset_href = item.assets["visual"].href
    
    print(f"Downloading true-color image to {output_filename}...")
    response = requests.get(asset_href, stream=True)
    response.raise_for_status()
    
    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("Download complete.")
    return output_filename

def upload_to_azure(file_path):
    """
    Uploads the downloaded GeoTIFF to Azure Blob Storage under coastal-tiles/incoming/.
    """
    print("Uploading to Azure Blob Storage...")
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set. Please create a .env file.")
        
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "coastal-tiles"
    blob_name = f"incoming/{os.path.basename(file_path)}"
    
    # Create container if it doesn't exist
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
        print(f"Created container '{container_name}'")
    except ResourceExistsError:
        pass # Container already exists
        
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    print(f"Uploading {file_path} to container '{container_name}' as '{blob_name}'...")
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        
    print("Upload complete!")

def main():
    try:
        # Step 1: Download the Sentinel-2 data from Planetary Computer
        tif_file = download_s2_data()
        
        # Step 2: Upload the downloaded file to Azure
        upload_to_azure(tif_file)
        
        print("\nProcess finished successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
