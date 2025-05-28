resource "aws_s3_bucket" "private_acl_auth_read_new_4" {
  bucket = "private-acl-auth-read-bucket-4-new"
}

resource "aws_s3_bucket_acl" "private_acl_auth_read_acl_new_4" {
  bucket = aws_s3_bucket.private_acl_auth_read_new_4.id
  acl    = "authenticated-read"
}