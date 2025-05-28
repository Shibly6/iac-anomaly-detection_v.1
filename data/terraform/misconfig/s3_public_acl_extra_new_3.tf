resource "aws_s3_bucket" "world_accessible_files" {
  bucket = "world-accessible-files-bucket"
}

resource "aws_s3_bucket_acl" "world_accessible_files_acl" {
  bucket = aws_s3_bucket.world_accessible_files.id
  acl    = "public-read"
}