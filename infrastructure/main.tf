terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# 1. Le Resource Group (Le "dossier" global qui contiendra tous tes services Azure)
resource "azurerm_resource_group" "rg" {
  name     = "rg-cyprus-fish-api"
  location = "East US" 
}

# 2. Le Service Plan (Définit la puissance de la machine. Ici : F1 = Plan Gratuit sous Linux)
resource "azurerm_service_plan" "plan" {
  name                = "plan-cyprus-fish-free"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  os_type             = "Linux"
  sku_name            = "F1"
}

# 3. L'Application Web (Le serveur qui va faire tourner ton image Docker)
resource "azurerm_linux_web_app" "api" {
  name                = "api-cyprus-fish-JayRay5" 
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_service_plan.plan.location
  service_plan_id     = azurerm_service_plan.plan.id

  site_config {
    always_on = false # Doit obligatoirement être à "false" pour le plan gratuit F1

    application_stack {
      # On configure Azure pour qu'il aille lire ton registre GitHub (GHCR)
      docker_registry_url = "https://ghcr.io"
      docker_image_name   = "jayray5/cyprus-fish-classifier-api:latest" # ⚠️ À adapter selon le nom de ton repo GitHub
    }
  }

  app_settings = {
    # Indique à Azure quel port ton conteneur Docker expose (on a mis EXPOSE 8000)
    "WEBSITES_PORT" = "8000"
    
    # Variable d'environnement pour FastAPI (CORS, mode, etc.)
    "ALLOWED_CORS_ORIGINS" = "*" 
  }
}

# 4. L'Output (Affiche l'URL publique de ton API dans ton terminal à la fin du déploiement)
output "api_url" {
  value       = "https://${azurerm_linux_web_app.api.name}.azurewebsites.net"
  description = "Public URL for the open source projet jayray5/cyprus-fish-recognition"
}