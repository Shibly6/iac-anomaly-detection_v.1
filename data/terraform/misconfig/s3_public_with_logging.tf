resource "aws_s3_bucket" "logs_bucket" {
  bucket = "logs-bucket"
}

resource "aws_s3_bucket" "public_with_logging" {
  bucket = "public-bucket-with-logging"
}

resource "aws_s3_bucket_acl" "public_with_logging_acl" {
  bucket = aws_s3_bucket.public_with_logging.id
  acl    = "public-read"
}

resource "aws_s3_bucket_logging" "example" {
  bucket        = aws_s3_bucket.public_with_logging.id
  target_bucket = aws_s3_bucket.logs_bucket.id
  target_prefix = "logs/"
}