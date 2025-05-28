resource "aws_s3_bucket" "public_encrypted" {
  bucket = "public-encrypted-bucket"
}

resource "aws_s3_bucket_acl" "public_encrypted_acl" {
  bucket = aws_s3_bucket.public_encrypted.id
  acl    = "public-read"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "example" {
  bucket = aws_s3_bucket.public_encrypted.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}