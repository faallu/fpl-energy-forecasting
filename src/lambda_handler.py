from src.pipeline.etl_pipeline import ETLPipeline

def handler(event, context):
    try:
        pipeline = ETLPipeline()
        success = pipeline.run_pipeline()

        return {
            'statusCode': 200,
            'body': {'message': 'ETL pipeline completed successfully'}
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': {'error': str(e)}
        }