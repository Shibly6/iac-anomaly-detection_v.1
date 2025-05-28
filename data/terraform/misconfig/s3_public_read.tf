resource "aws_s3_bucket" "public_read_explicit" {
  bucket = "public-read-bucket-explicit"
}

resource "aws_s3_bucket_acl" "public_read_explicit_acl" {
  bucket = aws_s3_bucket.public_read_explicit.id
  acl    = "public-read"
}