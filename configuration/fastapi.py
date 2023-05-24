from configuration.anonymous import AnonymousPathConfig


class FastApiConfig:
    def __init__(self):
        self.debug = True
        self.title = '中医四诊仪四诊合参服务'
        self.description = '四诊合参:基于望闻问切四诊表征，计算四诊合参结果。'
        self.version = '0.0.1.20240427'
        self.openapi_url = '/openapi.json'
        self.openapi_prefix = ''
        self.docs_url = '/docs'
        self.redoc_url = '/redoc'
        self.swagger_ui_oauth2_redirect_url = '/docs/oauth2-redirect'
        self.swagger_ui_init_oauth = None
        self.res_path = './res'
        self.request_log_to_mongo = True
        self.anonymousPathConfig: AnonymousPathConfig = AnonymousPathConfig()
