#logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(
    log_file: str = "pipeline.log", 
    level: int = logging.INFO,
    module: str = None
) -> logging.Logger:
    """
    Configura logging consistente para todo el proyecto.
    
    Args:
        log_file: Nombre del archivo de log
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        module: Nombre del módulo (si None, usa el nombre del caller)
    
    Returns:
        Logger configurado
    """
    # Crear directorio de logs si no existe
    log_path = Path("artifacts/logs") / log_file
    log_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Formato consistente
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configurar logger
    logger_name = module if module else __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Evitar handlers duplicados
    if not logger.handlers:
        # Handler para archivo
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Handler para consola (opcional)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger