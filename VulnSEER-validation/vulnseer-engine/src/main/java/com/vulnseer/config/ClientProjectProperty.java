package com.vulnseer.config;

public class ClientProjectProperty {

    private static ClassLoader instrumentedFuzzLoader; // instrumented client project classloader

    private static ClassLoader fuzzLoader; // not instrumented client project classloader

    public static ClassLoader getFuzzLoader() {
        return fuzzLoader;
    }

    public static void setFuzzLoader(ClassLoader fuzzLoader) {
        ClientProjectProperty.fuzzLoader = fuzzLoader;
    }

    public static ClassLoader getInstrumentedFuzzLoader() {
        return instrumentedFuzzLoader;
    }

    public static void setInstrumentedFuzzLoader(ClassLoader instrumentedFuzzLoader) {
        ClientProjectProperty.instrumentedFuzzLoader = instrumentedFuzzLoader;
    }
}
