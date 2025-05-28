resource "aws_s3_bucket" "private_acl_auth_read_new" {
  bucket = "private-acl-auth-read-bucket-new"
}

resource "aws_s3_bucket_acl" "private_acl_auth_read_acl_new" {
  bucket = aws_s3_bucket.private_acl_auth_read_new.id
  acl    = "authenticated-read"
}