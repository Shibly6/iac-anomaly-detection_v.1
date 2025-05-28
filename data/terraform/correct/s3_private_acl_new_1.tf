resource "aws_s3_bucket" "private_acl_1_new" {
  bucket = "private-acl-bucket-1-new"
}

resource "aws_s3_bucket_acl" "private_acl_1_acl_new" {
  bucket = aws_s3_bucket.private_acl_1_new.id
  acl    = "private"
}