# correct/s3_private_explicit.tf
resource "aws_s3_bucket" "private_explicit" {
  bucket = "private-bucket-explicit"
}

resource "aws_s3_bucket_acl" "private_explicit_acl" {
  bucket = aws_s3_bucket.private_explicit.id
  acl    = "private"
}