# correct/s3_private_default.tf
resource "aws_s3_bucket" "private_default" {
  bucket = "private-bucket-default"
  # No ACL specified (defaults to private)
}