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
}

# Resource Group
resource "azurerm_resource_group" "rg" {
  name     = "rg-chennai-coastal"
  location = "southindia"
}

# Storage Account
resource "azurerm_storage_account" "storage" {
  name                     = "coastaltiles"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
}

# Blob Container
resource "azurerm_storage_container" "container" {
  name                  = "coastal-tiles"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# Azure Functions Plan (Consumption)
resource "azurerm_service_plan" "plan" {
  name                = "coastal-plan"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  os_type             = "Linux"
  sku_name            = "Y1"
}

# Azure Functions App
resource "azurerm_linux_function_app" "func" {
  name                = "func-cloud-removal"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  service_plan_id     = azurerm_service_plan.plan.id
  storage_account_name = azurerm_storage_account.storage.name
  storage_account_access_key = azurerm_storage_account.storage.primary_access_key

  site_config {}
}
