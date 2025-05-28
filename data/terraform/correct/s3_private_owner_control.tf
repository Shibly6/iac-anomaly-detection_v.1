# correct/s3_private_owner_control.tf
resource "aws_s3_bucket" "private_owner_control" {
  bucket = "private-owner-control-bucket"
}

resource "aws_s3_bucket_acl" "private_owner_control_acl" {
  bucket = aws_s3_bucket.private_owner_control.id
  access_control_policy {
    grant {
      grantee {
        type = "CanonicalUser"
        id   = data.aws_canonical_user_id.current.id
      }
      permission = "FULL_CONTROL"
    }

    owner {
      id = data.aws_canonical_user_id.current.id
    }
  }
}

data "aws_canonical_user_id" "current" {}