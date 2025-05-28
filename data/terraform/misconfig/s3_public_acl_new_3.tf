resource "aws_s3_bucket" "anonymous_access_new" {
  bucket = "anonymous-access-new-bucket"
}

resource "aws_s3_bucket_acl" "anonymous_access_acl_new" {
  bucket = aws_s3_bucket.anonymous_access_new.id
  acl    = "public-read"
}