resource "aws_s3_bucket" "everyone_can_upload" {
  bucket = "everyone-can-upload-bucket"
}

resource "aws_s3_bucket_acl" "everyone_can_upload_acl" {
  bucket = aws_s3_bucket.everyone_can_upload.id
  acl    = "public-read-write"
}