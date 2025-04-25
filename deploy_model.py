from google.cloud import aiplatform

# Step 1: Initialize the Vertex AI environment
aiplatform.init(
    project="second-457818",
    location="us-central1"  # 'us-central1' is recommended
)

# Step 2: Upload the model to Vertex AI
model = aiplatform.Model.upload(
    display_name="bone-fracture-model",
    artifact_uri="gs://boneff",  # Folder containing your model file
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest"
)

# Step 3: Deploy the model to an endpoint
endpoint = model.deploy(
    deployed_model_display_name="bone-fracture-endpoint",
    machine_type="n1-standard-2"
)

# Step 4: Print endpoint details
print("✅ Model deployed successfully!")
print(f"🔗 Endpoint name: {endpoint.name}")
