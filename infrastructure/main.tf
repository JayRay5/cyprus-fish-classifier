terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "rg-terraform-meta"
    storage_account_name = "sttfstatejayray5cyprus"
    container_name       = "tfstate"
    key                  = "cyprus-fish-api.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# 1. Le Resource Group (Conservé en Canada Central)
resource "azurerm_resource_group" "rg" {
  name     = "rg-cyprus-fish-api"
  location = "Canada Central" 
}

# 2. Remplacement de l'App Service par Azure Container Instances (ACI)
resource "azurerm_container_group" "aci" {
  name                = "aci-cyprus-fish-api"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  ip_address_type     = "Public"
  dns_name_label      = "api-cyprus-fish-jayray5" # Ton sous-domaine unique (obligatoirement en minuscules)
  os_type             = "Linux"

  container {
    name   = "api"
    image  = "ghcr.io/jayray5/cyprus-fish-classifier-api:latest"
    cpu    = 1
    memory = 1.5 # 🟢 On alloue 1.5 Go de RAM pour que ton modèle ONNX tourne à l'aise sans crash

    ports {
      port     = 8000
      protocol = "TCP"
    }

    # Pour ACI, la configuration des variables d'environnement se fait via ce bloc
    environment_variables = {
      "WEBSITES_PORT"        = "8000"
      "ALLOWED_CORS_ORIGINS" = "*" 
      "API_SECRET_TOKEN"     = var.api_secret_token 
    }
  }
}

# 3. L'Output mis à jour (Génère dynamiquement l'URL publique de ton conteneur)
output "api_url" {
  value       = "http://${azurerm_container_group.aci.fqdn}:8000"
  description = "Public URL for the open source project jayray5/cyprus-fish-recognition"
}
variable "api_secret_token" {
  type        = string
  description = "Secret token to restrict users that request the API."
  sensitive   = true
}