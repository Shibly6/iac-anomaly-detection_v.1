resource "aws_s3_bucket" "open_documents_share" {
  bucket = "open-documents-share-bucket"
}

resource "aws_s3_bucket_acl" "open_documents_share_acl" {
  bucket = aws_s3_bucket.open_documents_share.id
  acl    = "public-read"
}