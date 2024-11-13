import shutil

def create_backup(data_file):
    backup_file = data_file + ".bak"  # Define the backup file name
    shutil.copy(data_file, backup_file)  # Copy the data file to the backup location
    print(f"Backup created: {backup_file}")

if __name__ == "__main__":
    # Example usage: replace 'your_data_file' with the actual data file path
    create_backup('data/T640_and_MET_data.pkl')
