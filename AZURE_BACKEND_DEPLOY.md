# Azure Deployment (TensorFlow Backend)

This guide deploys your heavy TensorFlow backend (`app.py`) to Azure App Service using a custom Docker image, then connects Vercel via proxy mode.

## 1) Prerequisites

- Azure CLI installed and logged in
- Docker not required locally (we use Azure Container Registry cloud build)
- Your repository contains:
  - `Dockerfile.azure`
  - `requirements-inference.txt`
  - `models/oral_cancer_model.h5`

## 2) Set Variables

Use Bash (Git Bash/WSL):

```bash
RG=oral-cancer-rg
LOCATION=eastus
ACR_NAME=oralcanceracr$RANDOM
PLAN=oral-cancer-plan
APP=oral-cancer-api-$RANDOM
IMAGE=oral-cancer-api:v1
```

## 3) Create Azure Resources

```bash
az login
az group create --name "$RG" --location "$LOCATION"
az acr create --resource-group "$RG" --name "$ACR_NAME" --sku Basic
az appservice plan create --name "$PLAN" --resource-group "$RG" --is-linux --sku B1
```

## 4) Build Container In ACR (Cloud Build)

```bash
az acr build --registry "$ACR_NAME" --image "$IMAGE" --file Dockerfile.azure .
```

## 5) Create Linux Web App From Container

```bash
az webapp create \
  --resource-group "$RG" \
  --plan "$PLAN" \
  --name "$APP" \
  --deployment-container-image-name "$ACR_NAME.azurecr.io/$IMAGE"
```

## 6) Grant App Service Pull Access To ACR

```bash
APP_ID=$(az webapp identity assign --name "$APP" --resource-group "$RG" --query principalId -o tsv)
ACR_ID=$(az acr show --name "$ACR_NAME" --resource-group "$RG" --query id -o tsv)
az role assignment create --assignee "$APP_ID" --scope "$ACR_ID" --role AcrPull
```

## 7) Configure Container Settings

```bash
az webapp config appsettings set \
  --resource-group "$RG" \
  --name "$APP" \
  --settings WEBSITES_PORT=8000
```

## 8) Restart And Test

```bash
az webapp restart --resource-group "$RG" --name "$APP"
BACKEND_URL="https://$APP.azurewebsites.net"
echo "$BACKEND_URL"
curl "$BACKEND_URL/api/health"
```

## 9) Connect Vercel To Azure Backend

In Vercel Project Settings -> Environment Variables:

- `INFERENCE_API_BASE_URL` = `https://<your-app-name>.azurewebsites.net`

Redeploy Vercel.

## 10) Verify End-To-End

- `https://<vercel-domain>/api/health` should show `mode: vercel-proxy`
- Upload an image from frontend and confirm real prediction response

## Troubleshooting

- If backend returns 503 model not loaded:
  - Confirm `models/oral_cancer_model.h5` exists in image build context
  - Check App Service logs:

```bash
az webapp log config --name "$APP" --resource-group "$RG" --docker-container-logging filesystem
az webapp log tail --name "$APP" --resource-group "$RG"
```

- If pull/image errors occur:
  - Re-run managed identity + `AcrPull` role assignment

- If cold starts are slow:
  - Move from B1 to a higher SKU (S1/P1v3)

