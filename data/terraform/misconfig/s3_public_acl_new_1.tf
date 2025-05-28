resource "aws_s3_bucket" "public_data_new" {
  bucket = "public-data-new-bucket"
}

resource "aws_s3_bucket_acl" "public_data_acl_new" {
  bucket = aws_s3_bucket.public_data_new.id
  acl    = "public-read"
}