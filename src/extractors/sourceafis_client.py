#!/usr/bin/env python3
"""
SourceAFIS JPype Client Wrapper.
Handles in-process Java Virtual Machine (JVM) initialization and calls to SourceAFIS library.
"""

import os
import sys
import base64
import logging
import subprocess
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Import JPype
try:
    import jpype
    import jpype.imports
    from jpype.types import JArray, JByte
    JPYPE_AVAILABLE = True
except ImportError:
    logger.error("JPype1 package is not installed. SourceAFIS integration will not function.")
    JPYPE_AVAILABLE = False


def find_jvm_dll(start_dir):
    """Recursively search for jvm.dll in the JRE directory."""
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.lower() == "jvm.dll":
                return os.path.join(root, file)
    return None


def get_system_java_version():
    """Returns system Java version or 0 if check fails."""
    try:
        res = subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = res.stderr
        if "version" in output:
            parts = output.splitlines()[0].split("\"")[1].split(".")
            if parts[0] == "1":
                return int(parts[1])
            else:
                return int(parts[0])
    except Exception:
        pass
    return 0


class SourceAFISClient:
    """
    SourceAFIS Java Library Wrapper using JPype for in-process JVM execution.
    """
    _jvm_started = False

    def __init__(self):
        self.enabled = False
        self.use_cache = True  # Enabled by default (fully verified for correctness)
        self._template_cache = {}
        if not JPYPE_AVAILABLE:
            logger.warning("SourceAFIS Client disabled: JPype1 is not installed.")
            return

        try:
            self._init_jvm()
            # Test imports to verify classpath loading
            from com.machinezoo.sourceafis import FingerprintImage, FingerprintTemplate, FingerprintMatcher, FingerprintImageOptions
            self.enabled = True
            logger.info("SourceAFIS JPype Client initialized successfully. Java classes loaded.")
        except Exception as e:
            logger.error(f"Failed to initialize SourceAFIS JPype JVM client: {e}", exc_info=True)
            self.enabled = False

    def _init_jvm(self):
        """Initializes the Java Virtual Machine with all dependencies on classpath."""
        if jpype.isJVMStarted():
            SourceAFISClient._jvm_started = True
            return

        # Resolve java_libs path
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_dir = os.path.join(current_dir, "java_libs")
        
        if not os.path.exists(lib_dir):
            raise FileNotFoundError(f"Java libraries directory not found at: {lib_dir}")

        # List all jars to construct explicit classpath
        jars = [os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if f.endswith(".jar")]
        if not jars:
            raise FileNotFoundError(f"No SourceAFIS JARs found in {lib_dir}. Run download_dependencies.py first.")

        # Determine JVM path to launch
        jvm_path = None
        system_version = get_system_java_version()
        
        if system_version < 9:
            # Need local JRE-17 jvm.dll
            local_jre_dir = os.path.join(lib_dir, "jre-17")
            if os.path.exists(local_jre_dir):
                jvm_dll = find_jvm_dll(local_jre_dir)
                if jvm_dll:
                    logger.info(f"Using local Java 17 JRE JVM binary: {jvm_dll}")
                    jvm_path = jvm_dll
                else:
                    logger.warning("Local Java 17 JRE folder found, but jvm.dll was not found recursively.")
            else:
                logger.warning(f"System Java version {system_version} < 9 and local JRE-17 was not found.")

        # Start the JVM
        if jvm_path:
            logger.info(f"Starting JVM (custom path) with classpath: {jars}")
            jpype.startJVM(jvm_path, classpath=jars)
        else:
            logger.info(f"Starting default system JVM with classpath: {jars}")
            jpype.startJVM(classpath=jars)

        SourceAFISClient._jvm_started = True

    def extract_template(self, img: np.ndarray, dpi: float = 500.0) -> str:
        """
        Extracts a SourceAFIS template from a raw grayscale NumPy fingerprint image.
        Returns the template as a Base64-encoded string representing the serialized binary template.
        """
        if not self.enabled:
            raise RuntimeError("SourceAFIS client is not initialized or failed to start.")

        if img is None or img.size == 0:
            raise ValueError("Fingerprint image is empty or invalid.")

        from com.machinezoo.sourceafis import FingerprintImage, FingerprintTemplate, FingerprintImageOptions

        # Standardize image to grayscale uint8
        if len(img.shape) == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        h, w = img.shape
        
        # Flatten image and convert to Java byte array (signed 8-bit bytes)
        flat_pixels = img.flatten().astype(np.int8)
        java_bytes = JArray(JByte)(flat_pixels.tolist())

        # Resolve and configure DPI (important for minutiae scaling)
        options = FingerprintImageOptions().dpi(float(dpi))

        # Instantiate Image and Template
        fp_image = FingerprintImage(w, h, java_bytes, options)
        template = FingerprintTemplate(fp_image)

        # Serialize template to Java byte[] and convert to Base64 Python string
        serialized_bytes = bytes(template.toByteArray())
        return base64.b64encode(serialized_bytes).decode("utf-8")

    def get_cached_template(self, template_b64: str):
        """
        Retrieves a cached Java FingerprintTemplate or creates and caches it.
        Bypasses caching if self.use_cache is False.
        """
        if not self.enabled or not template_b64:
            return None
            
        if not self.use_cache:
            return self.deserialize_template(template_b64)
            
        if template_b64 in self._template_cache:
            return self._template_cache[template_b64]
            
        template = self.deserialize_template(template_b64)
        if template is not None:
            self._template_cache[template_b64] = template
        return template

    def deserialize_template(self, template_b64: str):
        """
        Deserializes a base64-encoded string into a Java FingerprintTemplate object.
        """
        if not self.enabled or not template_b64:
            return None
        from com.machinezoo.sourceafis import FingerprintTemplate
        
        template_bytes = base64.b64decode(template_b64.encode("utf-8"))
        template_signed = np.frombuffer(template_bytes, dtype=np.int8)
        java_bytes = JArray(JByte)(template_signed.tolist())
        return FingerprintTemplate(java_bytes)

    def match(self, probe_template_b64: str, candidate_template_b64: str) -> float:
        """
        Matches a probe template against a candidate template.
        Returns the raw SourceAFIS similarity score (higher is more similar).
        """
        if not self.enabled:
            raise RuntimeError("SourceAFIS client is not initialized.")

        from com.machinezoo.sourceafis import FingerprintMatcher

        # Use caching wrapper to retrieve FingerprintTemplate objects
        probe_template = self.get_cached_template(probe_template_b64)
        cand_template = self.get_cached_template(candidate_template_b64)

        if probe_template is None or cand_template is None:
            return 0.0

        # Perform matching
        matcher = FingerprintMatcher(probe_template)
        return float(matcher.match(cand_template))

    def batch_match(self, probe_template_b64: str, candidate_templates_b64: list) -> list:
        """
        Performs high-performance batch matching of a single probe template against a list of candidates.
        Returns a list of raw matching scores.
        """
        if not self.enabled:
            raise RuntimeError("SourceAFIS client is not initialized.")

        from com.machinezoo.sourceafis import FingerprintMatcher

        # Use caching wrapper to retrieve probe template
        probe_template = self.get_cached_template(probe_template_b64)
        if probe_template is None:
            return [0.0] * len(candidate_templates_b64)

        # Initialize matcher once for the batch
        matcher = FingerprintMatcher(probe_template)

        scores = []
        for cand_b64 in candidate_templates_b64:
            if not cand_b64:
                scores.append(0.0)
                continue
            try:
                cand_template = self.get_cached_template(cand_b64)
                scores.append(float(matcher.match(cand_template)))
            except Exception as e:
                logger.error(f"Error matching candidate in batch: {e}")
                scores.append(0.0)

        return scores
