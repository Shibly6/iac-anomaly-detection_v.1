resource "aws_s3_bucket" "public_data_unrestricted" {
  bucket = "public-data-unrestricted-bucket"
}

resource "aws_s3_bucket_acl" "public_data_unrestricted_acl" {
  bucket = aws_s3_bucket.public_data_unrestricted.id
  acl    = "public-read-write"
}