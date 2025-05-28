resource "aws_s3_bucket" "open_bucket_new" {
  bucket = "open-bucket-example-new"
}

resource "aws_s3_bucket_acl" "open_bucket_acl_new" {
  bucket = aws_s3_bucket.open_bucket_new.id
  acl    = "public-read-write"
}