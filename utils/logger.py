"""
日志模块，记录项目运行日志
- 类： LoggerManager
- 对外统一接口： get_logger()
- 日志级别： DEBUG, INFO, WARNING, ERROR, CRITICAL
- 1个 logger 实例包含2个 handler:
    - file_handler: 负责将详细信息写入日志文件
    - console_handler: 负责将信息输出到控制台
"""

import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from .config import Config

class LoggerManager:
    """
    定义一个日志管理类，
    提供统一的日志配置和获取接口
    """
    # 类级别的单例实例引用，确保全局只创建一个 LoggerManager
    _instance = None
    # 实际的 logging.Logger 实例引用
    _logger = None

    def __new__(cls):
        """单例模式,确保全局只有一个日志管理器实例"""
        # 如果尚未创建实例，则调用父类 __new__ 创建一个新的实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        # 返回已存在或新创建的单例实例
        return cls._instance

    def __init__(self):
        """初始化日志管理器"""
        # 仅在首次初始化时配置日志记录器，避免重复配置
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        """配置日志记录器"""
        # 获取当前模块名对应的日志记录器对象（一般为模块级 logger）
        self._logger = logging.getLogger(__name__)
        # 设置日志级别为 DEBUG，记录尽可能详细的调试和运行信息
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        # 清空已有的日志处理器，防止重复添加导致重复输出
        self._logger.handlers = []
        # 创建文件日志处理器,支持并发写入且按大小滚动
        file_handler = ConcurrentRotatingFileHandler(
            Config.LOG_FILE,                # 指定日志文件路径，从配置类中读取
            maxBytes=Config.MAX_BYTES,      # 每个日志文件允许的最大字节数，到达上限会触发日志滚动
            backupCount=Config.BACKUP_COUNT, # 最多保留的历史备份日志文件数量
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)     
        
        # 定义日志输出格式：时间 - 日志器名称 - 级别 - 日志消息
        # %(name)s：日志器名称
        # %(filename)s：日志文件名
        # %(lineno)d：行号
        # %(message)s：日志消息
        # %(asctime)s：时间
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(filename)-18s:%(lineno)04d - %(levelname)-8s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # 配置控制台处理器 (Console Handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) 
        # 控制台日志格式
        console_formatter = logging.Formatter(
            "%(levelname)s - %(message)s",
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # ==========================================
        # 2.3 将两个处理器都添加到 Logger
        # ==========================================
        # 将配置好的处理器添加到日志记录器中
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    @property   # 属性化方法
    # _logger 私有变量，存着真正的日志对象
    # logger 
    ## 对外伪装成一个普通的属性（不用括号）
    ## 内部其实是一个方法（可以加逻辑）
    ## 只允许读，不允许写
    def logger(self):
        """获取日志记录器实例"""
        # 返回内部持有的 logging.Logger 对象
        return self._logger

    @classmethod
    def get_logger(cls):
        """类方法,获取日志记录器实例"""
        instance = cls()
        # 返回内部 logger，供业务代码直接使用
        return instance.logger


