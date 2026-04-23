terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
  resource_provider_registrations = "none"
}

# Use existing Resource Group
data "azurerm_resource_group" "rg" {
  name = "rg-chennai-coastal"
}

# Use existing Storage Account
data "azurerm_storage_account" "storage" {
  name                = "coastaltiles"
  resource_group_name = data.azurerm_resource_group.rg.name
}

# Blob Container (only create if it doesn't exist)
resource "azurerm_storage_container" "container" {
  name                  = "coastal-tiles"
  storage_account_id    = data.azurerm_storage_account.storage.id
  container_access_type = "private"
}

# Azure Functions Plan (only create if it doesn't exist)
resource "azurerm_service_plan" "plan" {
  name                = "coastal-plan"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  os_type             = "Linux"
  sku_name            = "Y1"
}

# Azure Functions App
resource "azurerm_linux_function_app" "func" {
  name                = "func-cloud-removal"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  service_plan_id     = azurerm_service_plan.plan.id
  storage_account_name = data.azurerm_storage_account.storage.name
  storage_account_access_key = data.azurerm_storage_account.storage.primary_access_key

  site_config {}
}
