resource "aws_s3_bucket" "private_acl_private_new_3" {
  bucket = "private-acl-private-bucket-3-new"
}

resource "aws_s3_bucket_acl" "private_acl_private_acl_new_3" {
  bucket = aws_s3_bucket.private_acl_private_new_3.id
  acl    = "private"
}