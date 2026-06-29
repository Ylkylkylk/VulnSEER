package com.vulnseer.fuzz;

import com.vulnseer.fuzz.instrument.FuzzClassTransformer;
import com.vulnseer.util.ClassUtil;
import com.vulnseer.util.IOUtil;
import com.vulnseer.util.URLUtil;

import java.io.IOException;
import java.io.InputStream;
import java.lang.instrument.IllegalClassFormatException;
import java.net.MalformedURLException;
import java.net.URLClassLoader;
import java.util.List;

public class FuzzClassLoader extends URLClassLoader {

    public static final String STATE_TABLE_CLASS_NAME = "com.vulnseer.instrument.state.GlobalStateTable";

    public static final String STATE_TABLE_INTERNAL_NAME = "com/vulnseer/instrument/state/GlobalStateTable";

    public static final String STATE_NODE_CLASS_NAME = "com.vulnseer.instrument.state.StateNode";

    public static final String STATE_NODE_INTERNAL_NAME = "com/vulnseer/instrument/state/StateNode";

    public static final String ADD_STATE_NODE_METHOD_NAME = "addStateNode";

    public static final String ADD_METHOD_SIGNATURE = "addMethodSignature";

    private static final byte[] STATE_TABLE_BYTES;

    private static final byte[] STATE_NODE_BYTES;

    private ClassLoader testcaseLoader = null;


    static {
        try {
            STATE_TABLE_BYTES = loadSupportClassBytes(STATE_TABLE_CLASS_NAME);
            STATE_NODE_BYTES = loadSupportClassBytes(STATE_NODE_CLASS_NAME);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private final FuzzClassTransformer transformer = new FuzzClassTransformer();

    public FuzzClassLoader(List<String> pathList) throws MalformedURLException {
        super(URLUtil.stringsToUrls(pathList.toArray(new String[0])), ClassLoader.getSystemClassLoader().getParent()); // break the parent delegation
    }

    public void setTestcaseLoader(ClassLoader testcaseLoader) {
        this.testcaseLoader = testcaseLoader;
    }

    private boolean shouldDelegateToFallbackLoader(String name) {
        return name.startsWith("org.slf4j.")
                || name.startsWith("ch.qos.logback.")
                || name.startsWith("org.apache.commons.logging.");
    }

    private Class<?> tryLoadFromFallbackLoaders(String name) throws ClassNotFoundException {
        ClassLoader[] fallbackLoaders = new ClassLoader[]{
                testcaseLoader,
                FuzzClassLoader.class.getClassLoader(),
                Thread.currentThread().getContextClassLoader(),
                ClassLoader.getSystemClassLoader()
        };

        for (ClassLoader loader : fallbackLoaders) {
            if (loader == null || loader == this) {
                continue;
            }
            try {
                return Class.forName(name, false, loader);
            } catch (ClassNotFoundException ignored) {
                // Try the next loader.
            }
        }

        throw new ClassNotFoundException("Cannot find class " + name);
    }

    private static byte[] loadSupportClassBytes(String className) throws IOException {
        String internalName = className.replace('.', '/') + ".class";
        ClassLoader[] candidateLoaders = new ClassLoader[]{
                FuzzClassLoader.class.getClassLoader(),
                Thread.currentThread().getContextClassLoader(),
                getSystemClassLoader()
        };

        for (ClassLoader loader : candidateLoaders) {
            if (loader == null) {
                continue;
            }

            byte[] bytes = ClassUtil.getClassBytes(loader, className);
            if (bytes != null) {
                return bytes;
            }
        }

        try (InputStream in = FuzzClassLoader.class.getResourceAsStream("/" + internalName)) {
            if (in != null) {
                return IOUtil.readBytes(in);
            }
        }

        throw new IOException("Cannot load support class bytes for " + className);
    }

    @Override
    public Class<?> findClass(String name) throws ClassNotFoundException {
        if (name == null) return null;

        if (shouldDelegateToFallbackLoader(name)) {
            return tryLoadFromFallbackLoaders(name);
        }

        if (name.startsWith("org.junit") || name.startsWith("junit")) { // Junit package is loaded by the Application ClassLoader
            return ClassLoader.getSystemClassLoader().loadClass(name);
        }

        if (STATE_TABLE_CLASS_NAME.equals(name)) {
            if (STATE_TABLE_BYTES == null) {
                throw new ClassNotFoundException("Missing support bytes for " + name);
            }
            return defineClass(name, STATE_TABLE_BYTES, 0, STATE_TABLE_BYTES.length);
        } else if (STATE_NODE_CLASS_NAME.equals(name)) {
            if (STATE_NODE_BYTES == null) {
                throw new ClassNotFoundException("Missing support bytes for " + name);
            }
            return defineClass(name, STATE_NODE_BYTES, 0, STATE_NODE_BYTES.length);
        } else {
            String internalName = name.replace('.', '/');
            String path = internalName.concat(".class");
            byte[] originalBytecode;

            try {
                InputStream in = super.getResourceAsStream(path);
                if (in == null && testcaseLoader != null) {
                    in = testcaseLoader.getResourceAsStream(path);
                }

                if (in == null) {
                    return tryLoadFromFallbackLoaders(name);
                }

                originalBytecode = IOUtil.readBytes(in);
            } catch (IOException e) {
                throw new ClassNotFoundException("I/O exception while loading class.", e);
            }

            assert (originalBytecode != null);

            byte[] bytesToLoad;
            try {
                byte[] instrumented = transformer.transform(this, internalName, null, null, originalBytecode);
                if (instrumented != null) {
                    bytesToLoad = instrumented;
                } else {
                    bytesToLoad = originalBytecode;
                }
            } catch (IllegalClassFormatException e) {
                bytesToLoad = originalBytecode;
            }

            return defineClass(name, bytesToLoad, 0, bytesToLoad.length);
        }
    }

}
