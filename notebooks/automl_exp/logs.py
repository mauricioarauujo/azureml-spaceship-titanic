from azureml.core import Workspace, Webservice
from azureml.exceptions import WebserviceException
import time

# Load the Azure ML workspace
ws = Workspace.from_config()

# Name of the web service
service_name = 'web-service-automl'

# Function to attempt to delete the service
def try_delete_service(service):
    try:
        service.delete()
        print("Web service deleted successfully.")
        return True
    except WebserviceException as e:
        print(f"Error deleting web service: {str(e)}")
        return False

# Get a reference to the web service
service = Webservice(workspace=ws, name=service_name)

# Print logs
print("Web service logs:")
print(service.get_logs())

# Wait for deployment operation to complete
try:
    service.wait_for_deployment()
except WebserviceException as e:
    # Ignore the exception related to the operation status
    if "No operation endpoint" not in str(e) and "Long running operation information not known" not in str(e):
        raise

# Attempt to delete the deployed web service with retries
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    if try_delete_service(service):
        break
    else:
        retry_count += 1
        print(f"Retrying deletion (attempt {retry_count}/{max_retries})...")
        time.sleep(10)  # Wait for 10 seconds before retrying

if retry_count == max_retries:
    print("Max retries reached. Unable to delete the web service.")
