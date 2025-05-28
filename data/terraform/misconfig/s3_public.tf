terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.0"
    }
  }
}

provider "aws" {
  access_key = "test"
  secret_key = "test"
  region     = "us-east-1"
  endpoints {
    s3 = "http://localhost:4566"
  }
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true
}

resource "aws_s3_bucket" "public_bucket" {
  bucket = "public-bucket"
  acl    = "public-read"
}
