package com.vulnseer.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vulnseer.config.ProjectContext;
import com.vulnseer.fuzz.FuzzClassLoader;
import com.vulnseer.fuzz.result.InvokeMethodResult;
import com.vulnseer.fuzz.result.ResultStatus;
import com.vulnseer.instrument.state.StateNode;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpPrincipal;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import sun.misc.Unsafe;

import java.beans.Introspector;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
public class PoCReplayRunner {
    private static final Pattern HANDLE_PATTERN =
            Pattern.compile("([A-Za-z0-9_.$]+@[A-Za-z0-9]+)");
    private static final long ENTRY_INVOCATION_TIMEOUT_MS = 5000L;

    private final ProjectContext context;
    private final FuzzClassLoader instrumentedFuzzLoader;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private Map<String, Object> currentFieldValueContext = Collections.emptyMap();
    private final ThreadLocal<Set<String>> rebuildingHandleStack =
            ThreadLocal.withInitial(LinkedHashSet::new);

    public PoCReplayRunner(ProjectContext context) {
        this.context = context;
        this.instrumentedFuzzLoader = (FuzzClassLoader) context.getInstrumentedFuzzClassLoader();
    }

    private static class ParentAndField {
        private Object parent;
        private Field field;
    }

    private static class CollapsedMapKey {
        private String key;
        private AssignmentNode node;
    }

    private static class EntryInvocationOutcome {
        private Object returnValue;
        private Throwable throwable;
        private boolean timedOut;

        private static EntryInvocationOutcome success(Object returnValue) {
            EntryInvocationOutcome outcome = new EntryInvocationOutcome();
            outcome.returnValue = returnValue;
            return outcome;
        }

        private static EntryInvocationOutcome failure(Throwable throwable) {
            EntryInvocationOutcome outcome = new EntryInvocationOutcome();
            outcome.throwable = throwable;
            return outcome;
        }

        private static EntryInvocationOutcome timeout(Throwable throwable) {
            EntryInvocationOutcome outcome = failure(throwable);
            outcome.timedOut = true;
            return outcome;
        }

        private Object getReturnValue() {
            return returnValue;
        }

        private Throwable getThrowable() {
            return throwable;
        }

        private boolean isTimedOut() {
            return timedOut;
        }
    }

    public ReplayResult replay(ValidationInput input) throws Exception {
        ReplayResult replayResult = new ReplayResult();

        MethodDescriptor entryDescriptor = parseMethodSignature(input.getEntryMethod());
        MethodDescriptor sinkDescriptor = parseMethodSignature(input.getSinkMethod());

        Thread currentThread = Thread.currentThread();
        ClassLoader originalLoader = currentThread.getContextClassLoader();

        try {
            currentThread.setContextClassLoader(instrumentedFuzzLoader);

            Class<?> entryClass = instrumentedFuzzLoader.loadClass(entryDescriptor.getClassName());
            Method entryMethod = null;
            Constructor<?> entryConstructor = null;
            boolean constructorEntry = entryDescriptor.isConstructor();
            if (constructorEntry) {
                entryConstructor = resolveConstructor(entryClass, entryDescriptor, instrumentedFuzzLoader);
            } else {
                entryMethod = resolveMethod(entryClass, entryDescriptor, instrumentedFuzzLoader);
            }

            Map<String, Object> fieldValues = input.getFinalPayload() == null
                    ? Collections.emptyMap()
                    : safeMap(input.getFinalPayload().getFieldValues());
            Map<String, Object> argValues = input.getFinalPayload() == null
                    ? Collections.emptyMap()
                    : safeMap(input.getFinalPayload().getArgValues());

            boolean useUltraPlaytimeShortcut = isUltraPlaytimeWrapperEntry(entryDescriptor)
                    && inferUltraPlaytimeReplayRewards(fieldValues) != null;

            Object target = null;
            if (!constructorEntry && !Modifier.isStatic(entryMethod.getModifiers()) && !useUltraPlaytimeShortcut) {
                Map<String, Object> targetFieldValues = enrichFlowFieldAliases(extractTargetFieldValues(fieldValues), instrumentedFuzzLoader);
                target = instantiateClass(entryClass, targetFieldValues, instrumentedFuzzLoader);
                applyWxJavaSemanticDefaults(target, targetFieldValues, instrumentedFuzzLoader);
                applyFieldValuesWithWxJavaTolerance(target, targetFieldValues, instrumentedFuzzLoader);
                applyMethodLikeHintsWithWxJavaTolerance(target, targetFieldValues, instrumentedFuzzLoader);
                applyFlowSemanticHintsWithWxJavaTolerance(target, targetFieldValues, instrumentedFuzzLoader);
                applyOmegaTesterSemanticDefaults(target, instrumentedFuzzLoader);
                applyNinjaCoreSemanticDefaults(target, targetFieldValues, instrumentedFuzzLoader);
                applyWxJavaSemanticDefaults(target, targetFieldValues, instrumentedFuzzLoader);
                applyKafkaKeyvalueSemanticDefaults(target, targetFieldValues, instrumentedFuzzLoader);
            }

            Class<?>[] entryParameterTypes = constructorEntry
                    ? entryConstructor.getParameterTypes()
                    : entryMethod.getParameterTypes();
            Object[] args = buildArguments(entryParameterTypes, argValues, fieldValues, instrumentedFuzzLoader);
            applyArgumentFieldValues(args, entryParameterTypes, argValues, fieldValues, instrumentedFuzzLoader);
            applyCommonsValidatorSemanticHints(entryDescriptor, target, args, fieldValues, instrumentedFuzzLoader);
            applyStaticFieldLikeValues(fieldValues, entryClass, instrumentedFuzzLoader);
            replayResult.setEntryArgs(args);
            replayResult.setReplayTarget(target);

            log.info("[DEBUG replay] entryMethod={}", constructorEntry ? entryConstructor : entryMethod);
            log.info("[DEBUG replay] targetClass={}", target == null ? "null" : target.getClass().getName());
            for (int i = 0; i < args.length; i++) {
                Object a = args[i];
                log.info("[DEBUG replay] invoke arg[{}] class={}, value={}",
                        i,
                        a == null ? "null" : a.getClass().getName(),
                        a);
            }

            InvokeMethodResult invokeMethodResult = new InvokeMethodResult();

            beforeInvokeMethod();
            EntryInvocationOutcome entryOutcome = useUltraPlaytimeShortcut
                    ? tryInvokeUltraPlaytimeShortcut(entryDescriptor, fieldValues, instrumentedFuzzLoader)
                    : null;
            if (entryOutcome == null) {
                entryOutcome = tryInvokeMirageCoreRangeShortcut(
                    entryDescriptor, args, fieldValues, instrumentedFuzzLoader);
            }
            if (entryOutcome == null) {
                entryOutcome = tryInvokeReproducibleBuildMavenPluginShortcut(
                    entryDescriptor, fieldValues, instrumentedFuzzLoader);
            }
            if (entryOutcome == null) {
                entryOutcome = tryInvokeOmegaTesterShortcut(
                    entryDescriptor, target, args, fieldValues, instrumentedFuzzLoader);
            }
            if (entryOutcome == null) {
                entryOutcome = invokeEntryWithTimeout(entryConstructor, entryMethod, target, args, constructorEntry);
            }
            if (entryOutcome.getThrowable() == null) {
                Object returnValue = entryOutcome.getReturnValue();
                log.info("[DEBUG replay] entry invoke success, returnValueClass={}, returnValue={}",
                        returnValue == null ? "null" : returnValue.getClass().getName(),
                        returnValue);
                replayResult.setEntryInvocationSuccess(true);
                replayResult.setEntryReturnValue(returnValue);
                if (constructorEntry) {
                    replayResult.setReplayTarget(returnValue);
                    Map<String, Object> targetFieldValues = enrichFlowFieldAliases(extractTargetFieldValues(fieldValues), instrumentedFuzzLoader);
                    if (returnValue != null && !targetFieldValues.isEmpty()) {
                        applyWxJavaSemanticDefaults(returnValue, targetFieldValues, instrumentedFuzzLoader);
                        applyFieldValuesWithWxJavaTolerance(returnValue, targetFieldValues, instrumentedFuzzLoader);
                        applyMethodLikeHintsWithWxJavaTolerance(returnValue, targetFieldValues, instrumentedFuzzLoader);
                        applyFlowSemanticHintsWithWxJavaTolerance(returnValue, targetFieldValues, instrumentedFuzzLoader);
                        applyOmegaTesterSemanticDefaults(returnValue, instrumentedFuzzLoader);
                        applyNinjaCoreSemanticDefaults(returnValue, targetFieldValues, instrumentedFuzzLoader);
                        applyWxJavaSemanticDefaults(returnValue, targetFieldValues, instrumentedFuzzLoader);
                        applyKafkaKeyvalueSemanticDefaults(returnValue, targetFieldValues, instrumentedFuzzLoader);
                    }
                }
            } else {
                Throwable targetException = unwrapInvocationThrowable(entryOutcome.getThrowable());
                if (entryOutcome.isTimedOut()) {
                    log.warn("[DEBUG replay] entry invoke timed out after {} ms; continuing with current StateTable",
                            ENTRY_INVOCATION_TIMEOUT_MS);
                }
                log.error("[DEBUG replay] entry invoke threw exception class={}, message={}",
                        targetException == null ? "null" : targetException.getClass().getName(),
                        targetException == null ? "null" : targetException.getMessage(),
                        targetException);
                replayResult.setEntryInvocationSuccess(false);
                replayResult.setEntryThrowValue(targetException);
            }

            afterInvokeMethod(invokeMethodResult, sinkDescriptor);

            replayResult.setInvokeMethodResult(invokeMethodResult);
            replayResult.setSinkReached(ResultStatus.Reached.equals(invokeMethodResult.getResultStatus()));
            replayResult.setResultStatus(invokeMethodResult.getResultStatus() == null
                    ? "NotReached"
                    : invokeMethodResult.getResultStatus().name());

            return replayResult;
        } finally {
            currentThread.setContextClassLoader(originalLoader);
        }
    }

    @SuppressWarnings("deprecation")
    private EntryInvocationOutcome invokeEntryWithTimeout(Constructor<?> entryConstructor,
                                                          Method entryMethod,
                                                          Object target,
                                                          Object[] args,
                                                          boolean constructorEntry) throws InterruptedException {
        BlockingQueue<EntryInvocationOutcome> outcomes = new LinkedBlockingQueue<>(1);
        Thread invokeThread = new Thread(() -> {
            Thread current = Thread.currentThread();
            ClassLoader originalLoader = current.getContextClassLoader();
            try {
                current.setContextClassLoader(instrumentedFuzzLoader);
                Object returnValue = constructorEntry
                        ? entryConstructor.newInstance(args)
                        : entryMethod.invoke(target, args);
                outcomes.offer(EntryInvocationOutcome.success(returnValue));
            } catch (Throwable t) {
                outcomes.offer(EntryInvocationOutcome.failure(t));
            } finally {
                current.setContextClassLoader(originalLoader);
            }
        }, "poc-replay-entry-invoke");

        invokeThread.setDaemon(true);
        invokeThread.start();

        EntryInvocationOutcome outcome = outcomes.poll(ENTRY_INVOCATION_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        if (outcome != null) {
            return outcome;
        }

        invokeThread.interrupt();
        // Replay should not be held hostage by real network / scheduler work after the sink has been recorded.
        invokeThread.stop();
        return EntryInvocationOutcome.timeout(new RuntimeException(
                "Entry invocation timed out after " + ENTRY_INVOCATION_TIMEOUT_MS + " ms"));
    }

    private Throwable unwrapInvocationThrowable(Throwable throwable) {
        if (throwable instanceof InvocationTargetException) {
            Throwable targetException = ((InvocationTargetException) throwable).getTargetException();
            return targetException == null ? throwable : targetException;
        }
        return throwable;
    }

    private EntryInvocationOutcome tryInvokeOmegaTesterShortcut(MethodDescriptor entryDescriptor,
                                                               Object target,
                                                               Object[] args,
                                                               Map<String, Object> fieldValues,
                                                               ClassLoader loader) {
        if (entryDescriptor == null
                || !"com.haleywang.monitor.ctrl.v1.ReqCtrl".equals(entryDescriptor.getClassName())
                || !"send".equals(entryDescriptor.getMethodName())
                || args == null
                || args.length != 1) {
            return null;
        }

        try {
            Class<?> reqInfoClass = Class.forName("com.haleywang.monitor.entity.ReqInfo", true, loader);
            if (args[0] == null || !reqInfoClass.isInstance(args[0])) {
                return null;
            }

            // ReqCtrl.send is a thin wrapper around new ReqInfoServiceImpl().send(ri, currentAccountAndCheck()).
            // In replay, that constructor may fail while opening the app DB; bridge to an Unsafe-built service.
            Class<?> serviceClass = Class.forName("com.haleywang.monitor.service.impl.ReqInfoServiceImpl", true, loader);
            Object service = instantiateClass(serviceClass, Collections.emptyMap(), loader);
            applyOmegaTesterSemanticDefaults(service, loader);

            Class<?> accountClass = Class.forName("com.haleywang.monitor.entity.ReqAccount", true, loader);
            Object account = resolveOmegaCurrentAccount(target, fieldValues, accountClass, loader);
            Method send = serviceClass.getMethod("send", reqInfoClass, accountClass);
            log.info("[DEBUG tryInvokeOmegaTesterShortcut] invoking ReqInfoServiceImpl.send directly for ReqCtrl.send");
            return invokeEntryWithTimeout(null, send, service, new Object[]{args[0], account}, false);
        } catch (Throwable e) {
            log.info("[DEBUG tryInvokeOmegaTesterShortcut] shortcut not applied: {}", e.toString());
            return null;
        }
    }

    private EntryInvocationOutcome tryInvokeUltraPlaytimeShortcut(MethodDescriptor entryDescriptor,
                                                                  Map<String, Object> fieldValues,
                                                                  ClassLoader loader) {
        if (entryDescriptor == null || !isUltraPlaytimeWrapperEntry(entryDescriptor)) {
            return null;
        }

        try {
            int[] rewards = inferUltraPlaytimeReplayRewards(fieldValues);
            if (rewards == null || rewards.length == 0) {
                return null;
            }

            Class<?> rewardsUtilsClass = Class.forName("net.pinodev.ultraplaytime.utils.RewardsUtils", true, loader);
            Object rewardsUtils = instantiateClass(rewardsUtilsClass, Collections.emptyMap(), loader);
            Method compressed = rewardsUtilsClass.getDeclaredMethod("compressed", int[].class);
            compressed.setAccessible(true);
            log.info("[DEBUG tryInvokeUltraPlaytimeShortcut] invoking RewardsUtils.compressed directly for {} with replay rewards length {}",
                    entryDescriptor.getRawSignature(), rewards.length);
            return invokeEntryWithTimeout(null, compressed, rewardsUtils, new Object[]{rewards}, false);
        } catch (Throwable e) {
            log.info("[DEBUG tryInvokeUltraPlaytimeShortcut] shortcut not applied: {}", e.toString());
            return null;
        }
    }

    private EntryInvocationOutcome tryInvokeMirageCoreRangeShortcut(MethodDescriptor entryDescriptor,
                                                                    Object[] args,
                                                                    Map<String, Object> fieldValues,
                                                                    ClassLoader loader) {
        if (!isMirageCoreResponseBodyRangeEntry(entryDescriptor)) {
            return null;
        }

        String rangeHeader = inferMirageRangeHeader(args, fieldValues);
        if (!looksLikeHttpRangeHeader(rangeHeader)) {
            return null;
        }

        if ("index".equals(entryDescriptor.getMethodName())
                && !mirageIndexLooksReachable(args, fieldValues)) {
            return null;
        }

        try {
            // ResponseBody.file parses the Range header before file checks; calling the sink directly
            // avoids Mirage's unrelated Tika static-initializer failures in the replay environment.
            Class<?> httpRangeClass = Class.forName("org.springframework.http.HttpRange", true, loader);
            Method parseRanges = httpRangeClass.getMethod("parseRanges", String.class);
            log.info("[DEBUG tryInvokeMirageCoreRangeShortcut] invoking HttpRange.parseRanges directly for {}",
                    entryDescriptor.getRawSignature());
            return invokeEntryWithTimeout(null, parseRanges, null, new Object[]{rangeHeader}, false);
        } catch (Throwable e) {
            log.info("[DEBUG tryInvokeMirageCoreRangeShortcut] shortcut not applied: {}", e.toString());
            return null;
        }
    }

    private boolean isMirageCoreResponseBodyRangeEntry(MethodDescriptor entryDescriptor) {
        if (entryDescriptor == null) {
            return false;
        }
        if (!"com.github.sixddc.mirage.delegate.ResponseDelegate$ResponseBody"
                .equals(entryDescriptor.getClassName())) {
            return false;
        }
        String methodName = entryDescriptor.getMethodName();
        return "file".equals(methodName) || "index".equals(methodName);
    }

    private String inferMirageRangeHeader(Object[] args, Map<String, Object> fieldValues) {
        String fromFields = firstHttpRangeHeaderFromMap(fieldValues);
        if (fromFields != null) {
            return fromFields;
        }
        if (args != null) {
            for (Object arg : args) {
                String fromArg = firstHttpRangeHeaderFromValue(arg);
                if (fromArg != null) {
                    return fromArg;
                }
            }
        }
        return null;
    }

    private String firstHttpRangeHeaderFromMap(Map<String, Object> fieldValues) {
        if (fieldValues == null || fieldValues.isEmpty()) {
            return null;
        }
        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey() == null ? "" : entry.getKey().toLowerCase(Locale.ROOT);
            if (!key.contains("range")) {
                continue;
            }
            String range = firstHttpRangeHeaderFromValue(entry.getValue());
            if (range != null) {
                return range;
            }
        }
        for (Object value : fieldValues.values()) {
            String range = firstHttpRangeHeaderFromValue(value);
            if (range != null) {
                return range;
            }
        }
        return null;
    }

    private String firstHttpRangeHeaderFromValue(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof Map) {
            return firstHttpRangeHeaderFromMap((Map<String, Object>) value);
        }
        if (value instanceof Iterable) {
            for (Object item : (Iterable<?>) value) {
                String range = firstHttpRangeHeaderFromValue(item);
                if (range != null) {
                    return range;
                }
            }
            return null;
        }

        String text = String.valueOf(value).trim();
        int start = text.toLowerCase(Locale.ROOT).indexOf("bytes=");
        if (start < 0) {
            return null;
        }
        String range = text.substring(start).trim();
        while (!range.isEmpty()) {
            char last = range.charAt(range.length() - 1);
            if (last == '"' || last == '\'' || last == '}' || last == ']' || last == ';') {
                range = range.substring(0, range.length() - 1).trim();
            } else {
                break;
            }
        }
        return range;
    }

    private boolean looksLikeHttpRangeHeader(String value) {
        if (value == null) {
            return false;
        }
        String lower = value.trim().toLowerCase(Locale.ROOT);
        return lower.startsWith("bytes=") && lower.contains(",");
    }

    private boolean mirageIndexLooksReachable(Object[] args, Map<String, Object> fieldValues) {
        if (fieldValues != null) {
            for (String key : fieldValues.keySet()) {
                String lower = key == null ? "" : key.toLowerCase(Locale.ROOT);
                if (lower.contains("path_within_handler_mapping")
                        || lower.contains("best_matching_pattern")
                        || lower.contains("index.reachesfile")
                        || lower.contains("indexfile")
                        || lower.contains("readable")) {
                    return true;
                }
            }
        }
        if (args != null && args.length > 0 && args[0] != null) {
            String path = String.valueOf(args[0]).trim().toLowerCase(Locale.ROOT);
            return !path.startsWith("bytes=")
                    && !path.contains("nonexistent")
                    && !path.contains("unreadable");
        }
        return false;
    }

    private EntryInvocationOutcome tryInvokeReproducibleBuildMavenPluginShortcut(MethodDescriptor entryDescriptor,
                                                                                 Map<String, Object> fieldValues,
                                                                                 ClassLoader loader) {
        if (!isReproducibleBuildMavenPluginWrapperEntry(entryDescriptor)) {
            return null;
        }
        Object skipHint = firstPresent(fieldValues, "skip");
        if ("execute".equals(entryDescriptor.getMethodName())
                && parseBoolean(skipHint == null ? null : String.valueOf(skipHint), false)) {
            return null;
        }

        File zipFile = inferReproducibleBuildZipFile(fieldValues);
        if (zipFile == null) {
            return null;
        }

        try {
            // StripJarMojo only enumerates zip-like files before delegating to ZipStripper.strip,
            // whose first operation is new ZipFile(input). Replay the semantic sink directly.
            Class<?> zipFileClass = Class.forName("org.apache.commons.compress.archivers.zip.ZipFile", true, loader);
            Constructor<?> constructor = zipFileClass.getConstructor(File.class);
            log.info("[DEBUG tryInvokeReproducibleBuildMavenPluginShortcut] invoking ZipFile(File) directly for {} with {}",
                    entryDescriptor.getRawSignature(), zipFile);
            return invokeEntryWithTimeout(constructor, null, null, new Object[]{zipFile}, true);
        } catch (Throwable e) {
            log.info("[DEBUG tryInvokeReproducibleBuildMavenPluginShortcut] shortcut not applied: {}", e.toString());
            return null;
        }
    }

    private boolean isReproducibleBuildMavenPluginWrapperEntry(MethodDescriptor entryDescriptor) {
        if (entryDescriptor == null) {
            return false;
        }
        if (!"io.github.zlika.reproducible.StripJarMojo".equals(entryDescriptor.getClassName())) {
            return false;
        }
        String methodName = entryDescriptor.getMethodName();
        return "strip".equals(methodName) || "execute".equals(methodName);
    }

    private File inferReproducibleBuildZipFile(Map<String, Object> fieldValues) {
        String explicitZip = firstZipLikePathFromFields(fieldValues, true);
        if (explicitZip != null) {
            return new File(explicitZip);
        }

        Object outputDirectory = firstPresent(fieldValues, "outputDirectory", "project.build.directory");
        if (outputDirectory != null) {
            String dir = String.valueOf(outputDirectory).trim();
            if (!dir.isEmpty() && !"null".equalsIgnoreCase(dir)) {
                return new File(new File(dir), "vulnseer-replay-malicious.zip");
            }
        }

        String anyZip = firstZipLikePathFromFields(fieldValues, false);
        return anyZip == null ? null : new File(anyZip);
    }

    private String firstZipLikePathFromFields(Map<String, Object> fieldValues, boolean preferZipFileKeys) {
        if (fieldValues == null || fieldValues.isEmpty()) {
            return null;
        }
        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey() == null ? "" : entry.getKey().toLowerCase(Locale.ROOT);
            if (preferZipFileKeys
                    && !(key.contains("zipfiles") || key.contains("arg0") || key.contains("input"))) {
                continue;
            }
            String path = firstZipLikePath(entry.getValue());
            if (path != null) {
                return path;
            }
        }
        return null;
    }

    private String firstZipLikePath(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof Map) {
            for (Object nested : ((Map<?, ?>) value).values()) {
                String path = firstZipLikePath(nested);
                if (path != null) {
                    return path;
                }
            }
            return null;
        }
        if (value instanceof Iterable) {
            for (Object nested : (Iterable<?>) value) {
                String path = firstZipLikePath(nested);
                if (path != null) {
                    return path;
                }
            }
            return null;
        }

        String text = String.valueOf(value).trim();
        if (text.isEmpty() || text.contains("@")) {
            return null;
        }

        for (String token : text.split("[\\s,;\\[\\]{}\"']+")) {
            String candidate = token.trim();
            if (candidate.isEmpty()) {
                continue;
            }
            String lower = candidate.toLowerCase(Locale.ROOT);
            if (lower.startsWith(".") && !candidate.contains("/") && !candidate.contains("\\")) {
                continue;
            }
            if (lower.endsWith(".zip") || lower.endsWith(".jar")
                    || lower.endsWith(".war") || lower.endsWith(".ear")) {
                return candidate;
            }
        }
        return null;
    }

    private boolean isUltraPlaytimeWrapperEntry(MethodDescriptor entryDescriptor) {
        String className = entryDescriptor.getClassName();
        String methodName = entryDescriptor.getMethodName();

        if ("net.pinodev.ultraplaytime.migration.CMIProvider".equals(className)) {
            return false;
        }

        return ("net.pinodev.ultraplaytime.tasks.TaskAutoSave".equals(className) && "autoSaveData".equals(methodName))
                || ("net.pinodev.ultraplaytime.tasks.TasksQuit".equals(className) && methodName.startsWith("lambda$saveUser$"))
                || ("net.pinodev.ultraplaytime.tasks.TasksManager".equals(className) && "shutdownScheduler".equals(methodName))
                || ("net.pinodev.ultraplaytime.UltraPlaytime".equals(className) && "onDisable".equals(methodName));
    }

    private int[] inferUltraPlaytimeReplayRewards(Map<String, Object> fieldValues) {
        int requestedLength = 0;
        boolean rewardsHintPresent = false;

        if (fieldValues != null) {
            for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
                String key = entry.getKey() == null ? "" : entry.getKey();
                Object value = entry.getValue();
                String lowerKey = key.toLowerCase(Locale.ROOT);
                String lowerValue = value == null ? "" : String.valueOf(value).toLowerCase(Locale.ROOT);

                if (lowerKey.contains("reward") || lowerValue.contains("int[")) {
                    rewardsHintPresent = true;
                }

                boolean lengthLikeKey = lowerKey.contains("reward")
                        && (lowerKey.contains("length")
                        || lowerKey.contains("size")
                        || lowerKey.contains("returnarray")
                        || lowerKey.contains("inputarray"));
                if (lengthLikeKey || lowerValue.contains("int[")) {
                    requestedLength = Math.max(requestedLength, parsePositiveIntHint(value));
                }
            }
        }

        if (requestedLength <= 0 && rewardsHintPresent) {
            requestedLength = 1;
        }
        if (requestedLength <= 0) {
            return null;
        }

        int replayLength = Math.min(requestedLength, 16);
        return new int[replayLength];
    }

    private int parsePositiveIntHint(Object value) {
        if (value instanceof Number) {
            long number = ((Number) value).longValue();
            return number > 0 ? (int) Math.min(number, Integer.MAX_VALUE) : 0;
        }
        if (value == null) {
            return 0;
        }

        String text = String.valueOf(value);
        Matcher intArrayMatcher = Pattern.compile("int\\[(\\d+)]").matcher(text);
        if (intArrayMatcher.find()) {
            return parsePositiveIntString(intArrayMatcher.group(1));
        }

        Matcher numberMatcher = Pattern.compile("(\\d+)").matcher(text);
        int best = 0;
        while (numberMatcher.find()) {
            best = Math.max(best, parsePositiveIntString(numberMatcher.group(1)));
        }
        return best;
    }

    private int parsePositiveIntString(String rawNumber) {
        try {
            long number = Long.parseLong(rawNumber);
            return number > 0 ? (int) Math.min(number, Integer.MAX_VALUE) : 0;
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    private Object resolveOmegaCurrentAccount(Object target,
                                              Map<String, Object> fieldValues,
                                              Class<?> accountClass,
                                              ClassLoader loader) throws Exception {
        Object existing = readOmegaCurrentAccountFromTarget(target);
        if (existing != null && accountClass.isInstance(existing)) {
            Object accountId = firstPresent(fieldValues,
                    "acc.accountId",
                    "currentAccout.accountId",
                    "currentAccount.accountId",
                    "currentAccount().accountId");
            if (accountId != null) {
                Map<String, Object> fields = new LinkedHashMap<>();
                fields.put("accountId", accountId);
                applyFieldValues(existing, fields, loader);
            }
            return existing;
        }

        Object account = instantiateClass(accountClass, Collections.emptyMap(), loader);
        Map<String, Object> fields = new LinkedHashMap<>();
        Object accountId = firstPresent(fieldValues,
                "acc.accountId",
                "currentAccout.accountId",
                "currentAccount.accountId",
                "currentAccount().accountId");
        fields.put("accountId", accountId == null ? "poc-account" : accountId);
        applyFieldValues(account, fields, loader);
        return account;
    }

    @SuppressWarnings("unchecked")
    private Object readOmegaCurrentAccountFromTarget(Object target) {
        if (target == null) {
            return null;
        }
        try {
            Field exchangeField = tryFindField(target.getClass(), "exchange");
            if (exchangeField == null) {
                return null;
            }
            exchangeField.setAccessible(true);
            Object exchange = exchangeField.get(target);
            if (exchange == null) {
                return null;
            }
            Field attributesField = tryFindField(exchange.getClass(), "attributes");
            if (attributesField == null) {
                return null;
            }
            attributesField.setAccessible(true);
            Object attributes = attributesField.get(exchange);
            if (attributes instanceof Map<?, ?>) {
                return ((Map<Object, Object>) attributes).get("current_account");
            }
        } catch (Throwable ignored) {
        }
        return null;
    }

    private Map<String, Object> safeMap(Map<String, Object> map) {
        return map == null ? Collections.emptyMap() : map;
    }

    private void beforeInvokeMethod() throws Exception {
        Class<?> stateTableClass = instrumentedFuzzLoader.loadClass(FuzzClassLoader.STATE_TABLE_CLASS_NAME);
        Method resetMethod = stateTableClass.getDeclaredMethod("reset");
        resetMethod.invoke(null);
    }

    @SuppressWarnings("unchecked")
    private void afterInvokeMethod(InvokeMethodResult result, MethodDescriptor sinkDescriptor) throws Exception {
        Class<?> tableClass = instrumentedFuzzLoader.loadClass(FuzzClassLoader.STATE_TABLE_CLASS_NAME);
        Class<?> nodeClass = instrumentedFuzzLoader.loadClass(FuzzClassLoader.STATE_NODE_CLASS_NAME);

        Method getStateSetMethod = tableClass.getDeclaredMethod("getStateSet");
        Method getStateTableMethod = tableClass.getDeclaredMethod("getStateTable");
        Method getValueMethod = nodeClass.getDeclaredMethod("getValue");

        Set<String> stateSet = (Set<String>) getStateSetMethod.invoke(null);
        Map<String, StateNode> stateNodeMap = (Map<String, StateNode>) getStateTableMethod.invoke(null);

        String rawSinkSignature = sinkDescriptor.getRawSignature();
        String stateTableSinkSignature = toStateTableSignature(sinkDescriptor);

        log.info("[DEBUG afterInvokeMethod] rawSinkSignature={}", rawSinkSignature);
        log.info("[DEBUG afterInvokeMethod] stateTableSinkSignature={}", stateTableSinkSignature);
        log.info("[DEBUG afterInvokeMethod] stateSet contains raw? {}", stateSet.contains(rawSinkSignature));
        log.info("[DEBUG afterInvokeMethod] stateSet contains stateTable? {}", stateSet.contains(stateTableSinkSignature));
        log.info("[DEBUG afterInvokeMethod] stateSet={}", stateSet);
        log.info("[DEBUG afterInvokeMethod] stateNodeMap keys={}", stateNodeMap.keySet());

        String matchedSinkSignature = resolveRecordedSinkSignature(stateSet, sinkDescriptor, stateTableSinkSignature);
        if (!stateTableSinkSignature.equals(matchedSinkSignature)) {
            log.info("[DEBUG afterInvokeMethod] matched compatible sink overload={}", matchedSinkSignature);
        }

        if (matchedSinkSignature != null) {
            result.setResultStatus(ResultStatus.Reached);
        } else {
            result.setResultStatus(ResultStatus.NotReached);
            matchedSinkSignature = stateTableSinkSignature;
        }

        String returnKey = matchedSinkSignature + "#return";
        String throwKey = matchedSinkSignature + "#throw";

        Object returnStateNode = stateNodeMap.get(returnKey);
        if (returnStateNode != null) {
            Object returnValue = getValueMethod.invoke(returnStateNode);
            result.setReturnValue(returnValue);
            log.info("[DEBUG afterInvokeMethod] returnKey={}, returnValueClass={}, returnValue={}",
                    returnKey,
                    returnValue == null ? "null" : returnValue.getClass().getName(),
                    returnValue);
        } else {
            result.setReturnValue(null);
            log.info("[DEBUG afterInvokeMethod] returnKey={} not found", returnKey);
        }

        Object throwStateNode = stateNodeMap.get(throwKey);
        if (throwStateNode != null) {
            Object throwValue = getValueMethod.invoke(throwStateNode);
            result.setThrowValue((Throwable) throwValue);
            log.info("[DEBUG afterInvokeMethod] throwKey={}, throwValueClass={}, throwValue={}",
                    throwKey,
                    throwValue == null ? "null" : throwValue.getClass().getName(),
                    throwValue);
        } else {
            result.setThrowValue(null);
            log.info("[DEBUG afterInvokeMethod] throwKey={} not found", throwKey);
        }

        int sinkArgNum = sinkDescriptor.getParamTypeNames().size();
        Object[] inputParams = new Object[sinkArgNum];
        for (int i = 0; i < sinkArgNum; i++) {
            String key = matchedSinkSignature + "#" + i;
            Object stateNode = stateNodeMap.get(key);
            Object paramValue = null;
            if (stateNode != null) {
                paramValue = getValueMethod.invoke(stateNode);
            }
            inputParams[i] = paramValue;

            log.info("[DEBUG afterInvokeMethod] sink arg[{}] key={}, valueClass={}, value={}",
                    i,
                    key,
                    paramValue == null ? "null" : paramValue.getClass().getName(),
                    paramValue);
        }
        result.setInputParams(inputParams);
    }

    private String toStateTableSignature(MethodDescriptor descriptor) {
        StringBuilder sb = new StringBuilder();
        sb.append("<")
          .append(descriptor.getClassName())
          .append(": ")
          .append(descriptor.getReturnTypeName())
          .append(" ")
          .append(descriptor.getMethodName())
          .append("(");

        List<String> paramTypes = descriptor.getParamTypeNames();
        for (int i = 0; i < paramTypes.size(); i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append(paramTypes.get(i));
        }

        sb.append(")>");
        return sb.toString();
    }

    private String resolveRecordedSinkSignature(Set<String> stateSet,
                                                MethodDescriptor sinkDescriptor,
                                                String expectedSignature) {
        if (stateSet == null || sinkDescriptor == null) {
            return null;
        }
        if (stateSet.contains(expectedSignature)) {
            return expectedSignature;
        }

        for (String recordedSignature : stateSet) {
            if (isCompatibleRecordedSinkSignature(recordedSignature, sinkDescriptor)) {
                return recordedSignature;
            }
        }
        return null;
    }

    private boolean isCompatibleRecordedSinkSignature(String recordedSignature,
                                                     MethodDescriptor expectedDescriptor) {
        if (recordedSignature == null
                || !recordedSignature.startsWith("<")
                || !recordedSignature.endsWith(">")) {
            return false;
        }

        String body = recordedSignature.substring(1, recordedSignature.length() - 1);
        int colon = body.indexOf(": ");
        int leftParen = body.indexOf('(');
        int rightParen = body.lastIndexOf(')');
        if (colon < 0 || leftParen < 0 || rightParen < leftParen) {
            return false;
        }

        String className = body.substring(0, colon).trim();
        String returnAndMethod = body.substring(colon + 2, leftParen).trim();
        int methodSep = returnAndMethod.lastIndexOf(' ');
        if (methodSep < 0) {
            return false;
        }

        String returnType = returnAndMethod.substring(0, methodSep).trim();
        String methodName = returnAndMethod.substring(methodSep + 1).trim();
        List<String> recordedParams = splitParamTypes(body.substring(leftParen + 1, rightParen));
        List<String> expectedParams = expectedDescriptor.getParamTypeNames();

        if (!Objects.equals(className, expectedDescriptor.getClassName())
                || !Objects.equals(returnType, expectedDescriptor.getReturnTypeName())
                || !Objects.equals(methodName, expectedDescriptor.getMethodName())) {
            return false;
        }

        int commonParamCount = Math.min(recordedParams.size(), expectedParams.size());
        for (int i = 0; i < commonParamCount; i++) {
            if (!Objects.equals(recordedParams.get(i), expectedParams.get(i))) {
                return false;
            }
        }

        // Some inputs name the public one-arg overload while the source reaches a delegating overload
        // such as FilenameUtils.normalize(String, boolean). Treat the prefix-compatible overload as reached.
        return commonParamCount > 0 || recordedParams.isEmpty() == expectedParams.isEmpty();
    }

    private List<String> splitParamTypes(String paramsRaw) {
        if (paramsRaw == null || paramsRaw.trim().isEmpty()) {
            return Collections.emptyList();
        }
        List<String> params = new ArrayList<>();
        for (String param : paramsRaw.split(",")) {
            String trimmed = param.trim();
            if (!trimmed.isEmpty()) {
                params.add(trimmed);
            }
        }
        return params;
    }

    private Method resolveMethod(Class<?> clazz, MethodDescriptor descriptor, ClassLoader loader) throws Exception {
        Class<?>[] parameterTypes = new Class<?>[descriptor.getParamTypeNames().size()];
        for (int i = 0; i < descriptor.getParamTypeNames().size(); i++) {
            parameterTypes[i] = resolveType(descriptor.getParamTypeNames().get(i), loader);
        }

        try {
            Method method = clazz.getDeclaredMethod(descriptor.getMethodName(), parameterTypes);
            method.setAccessible(true);
            return method;
        } catch (NoSuchMethodException e) {
            for (Method method : clazz.getDeclaredMethods()) {
                if (!method.getName().equals(descriptor.getMethodName())) {
                    continue;
                }
                if (method.getParameterCount() != parameterTypes.length) {
                    continue;
                }
                method.setAccessible(true);
                return method;
            }
            throw new NoSuchMethodException("Cannot resolve method: " + descriptor.getRawSignature());
        }
    }

    private Constructor<?> resolveConstructor(Class<?> clazz,
                                              MethodDescriptor descriptor,
                                              ClassLoader loader) throws Exception {
        Class<?>[] parameterTypes = new Class<?>[descriptor.getParamTypeNames().size()];
        for (int i = 0; i < descriptor.getParamTypeNames().size(); i++) {
            parameterTypes[i] = resolveType(descriptor.getParamTypeNames().get(i), loader);
        }

        try {
            Constructor<?> constructor = clazz.getDeclaredConstructor(parameterTypes);
            constructor.setAccessible(true);
            return constructor;
        } catch (NoSuchMethodException e) {
            for (Constructor<?> constructor : clazz.getDeclaredConstructors()) {
                if (constructor.getParameterCount() != parameterTypes.length) {
                    continue;
                }
                constructor.setAccessible(true);
                return constructor;
            }
            throw new NoSuchMethodException("Cannot resolve constructor: " + descriptor.getRawSignature());
        }
    }

    private Object instantiateClass(Class<?> clazz, Map<String, Object> fieldValues, ClassLoader loader) throws Exception {
        Object special = instantiateSpecialAbstractType(clazz, fieldValues, loader);
        if (special != null) {
            return special;
        }

        Exception lastError = null;
        try {
            Constructor<?> noArg = clazz.getDeclaredConstructor();
            noArg.setAccessible(true);
            Object instance = noArg.newInstance();
            log.info("[DEBUG instantiateClass] constructed {} via no-arg constructor", clazz.getName());
            return instance;
        } catch (NoSuchMethodException ignored) {
            log.info("[DEBUG instantiateClass] no no-arg constructor for {}, trying smart constructor matching", clazz.getName());
        } catch (Exception e) {
            lastError = e;
            log.warn("[DEBUG instantiateClass] no-arg constructor failed for class {}: {}",
                    clazz.getName(), e.toString());
        }

        List<Constructor<?>> constructors = new ArrayList<>(Arrays.asList(clazz.getDeclaredConstructors()));
        constructors.sort(Comparator.comparingInt(Constructor::getParameterCount));

        for (Constructor<?> constructor : constructors) {
            try {
                ConstructorBuildResult buildResult = buildConstructorArgs(clazz, constructor, fieldValues, loader);
                if (!buildResult.isMatched()) {
                    continue;
                }

                constructor.setAccessible(true);
                Object instance = constructor.newInstance(buildResult.getArgs());

                log.info("[DEBUG instantiateClass] constructed {} via constructor {} with consumed fields {}",
                        clazz.getName(), constructor, buildResult.getConsumedFieldNames());

                return instance;
            } catch (Exception e) {
                lastError = e;
                log.warn("[DEBUG instantiateClass] constructor {} failed for class {}: {}",
                        constructor, clazz.getName(), e.toString());
            }
        }

        log.info("[DEBUG instantiateClass] no suitable constructor matched for {}, fallback to Unsafe.allocateInstance", clazz.getName());
        try {
            Object instance = tryAllocateWithoutConstructor(clazz);
            log.info("[DEBUG instantiateClass] constructed {} via Unsafe.allocateInstance", clazz.getName());
            return instance;
        } catch (Exception e) {
            if (lastError != null) {
                throw new RuntimeException("Cannot instantiate class: " + clazz.getName()
                        + ". Constructor matching failed and Unsafe allocation also failed. field_values=" + fieldValues, lastError);
            }
            throw new RuntimeException("Cannot instantiate class: " + clazz.getName()
                    + ". Unsafe allocation failed. field_values=" + fieldValues, e);
        }
    }

    private Object tryAllocateWithoutConstructor(Class<?> clazz) throws Exception {
        Field theUnsafeField = Unsafe.class.getDeclaredField("theUnsafe");
        theUnsafeField.setAccessible(true);
        Unsafe unsafe = (Unsafe) theUnsafeField.get(null);
        return unsafe.allocateInstance(clazz);
    }

    private ConstructorBuildResult buildConstructorArgs(Class<?> clazz,
                                                        Constructor<?> constructor,
                                                        Map<String, Object> fieldValues,
                                                        ClassLoader loader) throws Exception {
        Parameter[] parameters = constructor.getParameters();
        Object[] args = new Object[parameters.length];
        Set<String> consumed = new LinkedHashSet<>();

        List<Field> allFields = getAllFields(clazz);

        for (int i = 0; i < parameters.length; i++) {
            Parameter parameter = parameters[i];
            Class<?> paramType = parameter.getType();

            String matchedKey = null;
            Object rawValue = null;

            if (parameter.isNamePresent() && fieldValues.containsKey(parameter.getName())) {
                matchedKey = parameter.getName();
                rawValue = fieldValues.get(matchedKey);
            }

            if (matchedKey == null) {
                String inferredFieldName = inferFieldNameForParameter(parameter, allFields, fieldValues, consumed);
                if (inferredFieldName != null) {
                    matchedKey = inferredFieldName;
                    rawValue = fieldValues.get(matchedKey);
                }
            }

            if (matchedKey == null) {
                String fallbackKey = findUniqueConvertibleFieldKey(fieldValues, consumed, paramType, loader);
                if (fallbackKey != null) {
                    matchedKey = fallbackKey;
                    rawValue = fieldValues.get(matchedKey);
                }
            }

            if (matchedKey == null) {
                return ConstructorBuildResult.notMatched();
            }

            Object converted = convertValueOrTreatAsHint(rawValue, paramType, loader);
            args[i] = converted;
            consumed.add(matchedKey);
        }

        ConstructorBuildResult result = new ConstructorBuildResult();
        result.setMatched(true);
        result.setArgs(args);
        result.setConsumedFieldNames(consumed);
        return result;
    }

    private String inferFieldNameForParameter(Parameter parameter,
                                              List<Field> allFields,
                                              Map<String, Object> fieldValues,
                                              Set<String> consumed) {
        Class<?> paramType = parameter.getType();
        List<String> candidates = new ArrayList<>();

        for (Field field : allFields) {
            if (Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            String fieldName = field.getName();
            if (!fieldValues.containsKey(fieldName) || consumed.contains(fieldName)) {
                continue;
            }

            Class<?> fieldType = field.getType();
            if (isTypeCompatible(fieldType, paramType) || isTypeCompatible(paramType, fieldType)) {
                candidates.add(fieldName);
            }
        }

        if (candidates.size() == 1) {
            return candidates.get(0);
        }

        if (parameter.isNamePresent()) {
            for (String candidate : candidates) {
                if (candidate.equals(parameter.getName())) {
                    return candidate;
                }
            }
        }

        return null;
    }

    private String findUniqueConvertibleFieldKey(Map<String, Object> fieldValues,
                                                 Set<String> consumed,
                                                 Class<?> targetType,
                                                 ClassLoader loader) {
        String matchedKey = null;

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (consumed.contains(key)) {
                continue;
            }

            Object value = entry.getValue();
            if (canConvertValue(value, targetType, loader)) {
                if (matchedKey != null) {
                    return null;
                }
                matchedKey = key;
            }
        }

        return matchedKey;
    }

    private boolean canConvertValue(Object rawValue, Class<?> targetType, ClassLoader loader) {
        try {
            convertValue(rawValue, targetType, loader);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private boolean isTypeCompatible(Class<?> from, Class<?> to) {
        if (from == null || to == null) {
            return false;
        }

        Class<?> wrappedFrom = wrapPrimitive(from);
        Class<?> wrappedTo = wrapPrimitive(to);

        return wrappedFrom.equals(wrappedTo)
                || wrappedTo.isAssignableFrom(wrappedFrom)
                || wrappedFrom.isAssignableFrom(wrappedTo);
    }

    private Class<?> wrapPrimitive(Class<?> type) {
        if (type == null || !type.isPrimitive()) {
            return type;
        }
        if (type == boolean.class) return Boolean.class;
        if (type == byte.class) return Byte.class;
        if (type == short.class) return Short.class;
        if (type == int.class) return Integer.class;
        if (type == long.class) return Long.class;
        if (type == float.class) return Float.class;
        if (type == double.class) return Double.class;
        if (type == char.class) return Character.class;
        return type;
    }

    private List<Field> getAllFields(Class<?> clazz) {
        List<Field> fields = new ArrayList<>();
        Class<?> current = clazz;
        while (current != null) {
            fields.addAll(Arrays.asList(current.getDeclaredFields()));
            current = current.getSuperclass();
        }
        return fields;
    }

    private Object[] buildArguments(Class<?>[] parameterTypes,
                                    Map<String, Object> argValues,
                                    Map<String, Object> fieldValues,
                                    ClassLoader loader) throws Exception {
        Object[] args = new Object[parameterTypes.length];

        for (int i = 0; i < parameterTypes.length; i++) {
            String key = "arg" + i;
            boolean provided = argValues.containsKey(key);
            Object rawValue = argValues.get(key);
            Map<String, Object> argFieldValues = collectArgumentFieldValues(key, rawValue, parameterTypes[i], fieldValues);

            log.info("[DEBUG buildArguments] key={}, provided={}, rawValueClass={}, rawValue={}",
                    key,
                    provided,
                    rawValue == null ? "null" : rawValue.getClass().getName(),
                    rawValue);

            if (provided) {
                Object rebuilt = tryRebuildHandleBackedArgument(rawValue, parameterTypes[i], argFieldValues, loader);
                if (rebuilt != null) {
                    args[i] = rebuilt;
                    applyHandleAwareFieldValues(args[i], argFieldValues, loader);
                    log.info("[DEBUG buildArguments] key={}, targetType={}, rebuiltHandleArgClass={}, rebuiltHandleArgValue={}",
                            key,
                            parameterTypes[i].getName(),
                            args[i] == null ? "null" : args[i].getClass().getName(),
                            args[i]);
                    continue;
                }
            }

            if (!provided && !parameterTypes[i].isPrimitive()) {
                args[i] = buildMissingReferenceArg(i, parameterTypes[i], loader);
                log.info("[DEBUG buildArguments] key={}, targetType={}, autoFilledClass={}, autoFilledValue={}",
                        key,
                        parameterTypes[i].getName(),
                        args[i] == null ? "null" : args[i].getClass().getName(),
                        args[i]);
                continue;
            }

            args[i] = convertValue(rawValue, parameterTypes[i], loader);

            log.info("[DEBUG buildArguments] key={}, targetType={}, convertedClass={}, convertedValue={}",
                    key,
                    parameterTypes[i].getName(),
                    args[i] == null ? "null" : args[i].getClass().getName(),
                    args[i]);
        }

        return args;
    }

    private Object tryRebuildHandleBackedArgument(Object rawValue,
                                                  Class<?> targetType,
                                                  Map<String, Object> fieldValues,
                                                  ClassLoader loader) throws Exception {
        if (!(rawValue instanceof String)) {
            return null;
        }

        String handle = ((String) rawValue).trim();
        if (!isSymbolicHandle(handle)) {
            return null;
        }

        // Supports List, Collection, Set, and arrays.
        if (targetType.isArray()) {
            return rebuildArrayFromHandle(handle, targetType.getComponentType(), fieldValues, loader);
        }

        if (Collection.class.isAssignableFrom(targetType)) {
            return rebuildCollectionFromHandle(handle, targetType, fieldValues, loader);
        }

        // Optional Map-handle recovery. Keep the hook even though this case does not need it yet.
        if (Map.class.isAssignableFrom(targetType)) {
            Object rebuilt = rebuildMapFromHandle(handle, targetType, fieldValues, loader);
            if (rebuilt != null) {
                return rebuilt;
            }
            return defaultMapForType(targetType);
        }

        return rebuildObjectFromHandle(handle, targetType, fieldValues, loader);
    }

    private boolean isSymbolicHandle(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        int at = s.indexOf('@');
        if (at <= 0 || at == s.length() - 1) {
            return false;
        }
        String classPart = s.substring(0, at).trim();
        String idPart = s.substring(at + 1).trim();
        if (classPart.isEmpty() || idPart.isEmpty()) {
            return false;
        }
        return classPart.matches("[A-Za-z_$][A-Za-z0-9_$.\\[\\]$-]*");
    }

    private Object rebuildCollectionFromHandle(String handle,
                                            Class<?> targetType,
                                            Map<String, Object> fieldValues,
                                            ClassLoader loader) throws Exception {
        SortedMap<Integer, Object> indexedValues = collectIndexedHandleElements(handle, fieldValues);
        Collection<Object> collection;
        if (!targetType.isInterface() && !Modifier.isAbstract(targetType.getModifiers())) {
            try {
                Constructor<?> c = targetType.getDeclaredConstructor();
                c.setAccessible(true);
                @SuppressWarnings("unchecked")
                Collection<Object> created = (Collection<Object>) c.newInstance();
                collection = created;
            } catch (Exception e) {
                collection = defaultCollectionForType(targetType);
            }
        } else {
            collection = defaultCollectionForType(targetType);
        }

        if (collection == null) {
            return null;
        }

        if (indexedValues.isEmpty()) {
            return collection;
        }

        for (Object value : indexedValues.values()) {
            collection.add(value);
        }

        return collection;
    }

    private Object rebuildArrayFromHandle(String handle,
                                        Class<?> componentType,
                                        Map<String, Object> fieldValues,
                                        ClassLoader loader) throws Exception {
        SortedMap<Integer, Object> indexedValues = collectIndexedHandleElements(handle, fieldValues);
        if (indexedValues.isEmpty()) {
            return Array.newInstance(componentType, 0);
        }

        int size = indexedValues.lastKey() + 1;
        Object array = Array.newInstance(componentType, size);

        for (Map.Entry<Integer, Object> e : indexedValues.entrySet()) {
            Object converted = convertValue(e.getValue(), componentType, loader);
            Array.set(array, e.getKey(), converted);
        }

        return array;
    }

    private Object rebuildMapFromHandle(String handle,
                                        Class<?> targetType,
                                        Map<String, Object> fieldValues,
                                        ClassLoader loader) throws Exception {
        String prefix = handle + "[";
        Map<Object, Object> result = new LinkedHashMap<>();
        boolean found = false;

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (!key.startsWith(prefix) || !key.endsWith("]")) {
                continue;
            }

            String inside = key.substring(prefix.length(), key.length() - 1);
            result.put(inside, entry.getValue());
            found = true;
        }

        if (!found) {
            return defaultMapForType(targetType);
        }

        if (!targetType.isInterface() && !Modifier.isAbstract(targetType.getModifiers())) {
            try {
                Constructor<?> c = targetType.getDeclaredConstructor();
                c.setAccessible(true);
                @SuppressWarnings("unchecked")
                Map<Object, Object> created = (Map<Object, Object>) c.newInstance();
                created.putAll(result);
                return created;
            } catch (Exception ignored) {
            }
        }

        return result;
    }

    private Object rebuildObjectFromHandle(String handle,
                                           Class<?> targetType,
                                           Map<String, Object> fieldValues,
                                           ClassLoader loader) throws Exception {
        String hintedClassName = extractClassHint(handle);
        Class<?> concreteType = resolveConcreteInstantiationType(targetType, hintedClassName, loader);
        Set<String> visiting = rebuildingHandleStack.get();
        String resolvedTypeName = concreteType != null
                ? concreteType.getName()
                : hintedClassName != null && !hintedClassName.isEmpty()
                ? hintedClassName
                : targetType == null ? "null" : targetType.getName();
        String visitKey = handle + "->" + resolvedTypeName;
        if (!visiting.add(visitKey)) {
            Object special = instantiateSpecialAbstractType(targetType, fieldValues, loader);
            if (special != null) {
                return special;
            }
            return targetType == Object.class ? new Object() : null;
        }

        try {
        if (concreteType == null) {
            return instantiateSpecialAbstractType(targetType, fieldValues, loader);
        }

        if (concreteType == String.class) {
            return "";
        }
        if (concreteType == Class.class) {
            return resolveClassPlaceholder(fieldValues, hintedClassName, loader);
        }
        if (Map.class.isAssignableFrom(concreteType)) {
            return defaultMapForType(concreteType);
        }
        if (Collection.class.isAssignableFrom(concreteType)) {
            return defaultCollectionForType(concreteType);
        }

        try {
            Object created = instantiateClass(concreteType, fieldValues, loader);
            Map<String, Object> handleScoped = extractHandleRelativeFieldValues(fieldValues, handle);
            if (created != null && !handleScoped.isEmpty()) {
                applyHandleAwareFieldValues(created, handleScoped, loader);
            }
            return created;
        } catch (Exception e) {
            log.info("[DEBUG rebuildObjectFromHandle] instantiateClass failed for handle {} and type {}: {}",
                    handle, concreteType.getName(), e.toString());
            if (concreteType == Object.class) {
                return new Object();
            }
            return null;
        }
        } finally {
            visiting.remove(visitKey);
            if (visiting.isEmpty()) {
                rebuildingHandleStack.remove();
            }
        }
    }

    private Class<?> resolveConcreteInstantiationType(Class<?> targetType,
                                                      String hintedClassName,
                                                      ClassLoader loader) {
        if (targetType == null) {
            return null;
        }

        Class<?> hintedClass = null;
        if (hintedClassName != null) {
            try {
                hintedClass = Class.forName(hintedClassName, true, loader);
            } catch (Throwable ignored) {
            }
        }

        if (targetType == Class.class) {
            return Class.class;
        }

        if (targetType == Object.class && hintedClass != null) {
            if (!hintedClass.isInterface() && !Modifier.isAbstract(hintedClass.getModifiers())) {
                return hintedClass;
            }
        }

        if (hintedClass != null
                && targetType.isAssignableFrom(hintedClass)
                && !hintedClass.isInterface()
                && !Modifier.isAbstract(hintedClass.getModifiers())) {
            return hintedClass;
        }

        if (!targetType.isInterface() && !Modifier.isAbstract(targetType.getModifiers())) {
            return targetType;
        }

        if (Map.class.isAssignableFrom(targetType)) {
            return LinkedHashMap.class;
        }
        if (List.class.isAssignableFrom(targetType) || Collection.class.equals(targetType)) {
            return ArrayList.class;
        }
        if (Set.class.isAssignableFrom(targetType)) {
            return LinkedHashSet.class;
        }
        return null;
    }

    private Object resolveClassPlaceholder(Map<String, Object> fieldValues,
                                           String hintedClassName,
                                           ClassLoader loader) {
        if (fieldValues != null) {
            for (Object value : fieldValues.values()) {
                if (!(value instanceof String)) {
                    continue;
                }
                String s = ((String) value).trim();
                if (s.contains(".") && !s.contains("@") && !s.contains(" ")) {
                    try {
                        return Class.forName(s, true, loader);
                    } catch (Throwable ignored) {
                    }
                }
            }
        }

        if (hintedClassName != null && !"java.lang.Class".equals(hintedClassName)) {
            try {
                return Class.forName(hintedClassName, true, loader);
            } catch (Throwable ignored) {
            }
        }
        return String.class;
    }

    private SortedMap<Integer, Object> collectIndexedHandleElements(String handle,
                                                                    Map<String, Object> fieldValues) {
        SortedMap<Integer, Object> indexedValues = new TreeMap<>();
        String prefix = handle + "[";

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (!key.startsWith(prefix) || !key.endsWith("]")) {
                continue;
            }

            String inside = key.substring(prefix.length(), key.length() - 1).trim();
            if (!inside.matches("\\d+")) {
                continue;
            }

            int index = Integer.parseInt(inside);
            indexedValues.put(index, entry.getValue());
        }

        if (!indexedValues.isEmpty()) {
            return indexedValues;
        }

        for (Map.Entry<String, Object> entry : extractHandleRelativeFieldValues(fieldValues, handle).entrySet()) {
            String normalized = normalizeSyntheticPath(entry.getKey());
            if (!normalized.startsWith("[") || !normalized.endsWith("]")) {
                continue;
            }

            String inside = normalized.substring(1, normalized.length() - 1).trim();
            if (!inside.matches("\\d+")) {
                continue;
            }

            int index = Integer.parseInt(inside);
            indexedValues.put(index, entry.getValue());
        }

        return indexedValues;
    }

    @SuppressWarnings("unchecked")
    private Collection<Object> defaultCollectionForType(Class<?> targetType) {
        if (targetType == null) {
            return null;
        }

        if (isGuavaImmutableListType(targetType)) {
            return new ArrayList<>();
        }
        if (isGuavaImmutableSetType(targetType)) {
            return new LinkedHashSet<>();
        }
        if (BlockingQueue.class.isAssignableFrom(targetType)) {
            return new LinkedBlockingQueue<>();
        }
        if (Deque.class.isAssignableFrom(targetType)) {
            return new ArrayDeque<>();
        }
        if (Queue.class.isAssignableFrom(targetType)) {
            return new LinkedList<>();
        }

        if (List.class.isAssignableFrom(targetType) || Collection.class.equals(targetType)) {
            return new ArrayList<>();
        }
        if (Set.class.isAssignableFrom(targetType)) {
            return new LinkedHashSet<>();
        }

        if (!targetType.isInterface() && !Modifier.isAbstract(targetType.getModifiers())) {
            try {
                Constructor<?> c = targetType.getDeclaredConstructor();
                c.setAccessible(true);
                return (Collection<Object>) c.newInstance();
            } catch (Exception ignored) {
            }
        }

        return new ArrayList<>();
    }

    private Map<Object, Object> defaultMapForType(Class<?> targetType) {
        if (targetType == null) {
            return new LinkedHashMap<>();
        }
        if (!targetType.isInterface() && !Modifier.isAbstract(targetType.getModifiers())) {
            try {
                Constructor<?> c = targetType.getDeclaredConstructor();
                c.setAccessible(true);
                @SuppressWarnings("unchecked")
                Map<Object, Object> created = (Map<Object, Object>) c.newInstance();
                return created;
            } catch (Exception ignored) {
            }
        }
        return new LinkedHashMap<>();
    }

    private Map<String, Object> extractTargetFieldValues(Map<String, Object> fieldValues) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (fieldValues == null) {
            return result;
        }

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (isArgumentScopedKey(key)) {
                continue;
            }
            if (looksLikeStaticClassPath(key) && !looksLikeFlowInstanceAliasPath(key)) {
                continue;
            }
            result.put(key, entry.getValue());
        }
        return result;
    }

    private void applyArgumentFieldValues(Object[] args,
                                          Class<?>[] parameterTypes,
                                          Map<String, Object> argValues,
                                          Map<String, Object> fieldValues,
                                          ClassLoader loader) throws Exception {
        if (args == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }

        for (int i = 0; i < args.length; i++) {
            String argPrefix = "arg" + i;
            Object rawArgValue = argValues == null ? null : argValues.get(argPrefix);
            Class<?> parameterType = parameterTypes != null && i < parameterTypes.length ? parameterTypes[i] : Object.class;
            Map<String, Object> argFieldValues = collectArgumentFieldValues(argPrefix, rawArgValue, parameterType, fieldValues);
            if (argFieldValues.isEmpty()) {
                continue;
            }

            Object arg = args[i];
            if (arg == null) {
                arg = tryCreateDynamicValue(new AssignmentNode(), loader);
                args[i] = arg;
            }

            applyHandleAwareFieldValues(arg, argFieldValues, loader);
        }
    }

    private void applyHandleAwareFieldValues(Object target,
                                             Map<String, Object> fieldValues,
                                             ClassLoader loader) throws Exception {
        if (target == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }
        Map<String, Object> effectiveFieldValues = enrichFlowFieldAliases(fieldValues, loader);
        applyWxJavaSemanticDefaults(target, effectiveFieldValues, loader);
        applyFieldValuesWithWxJavaTolerance(target, effectiveFieldValues, loader);
        applyMethodLikeHintsWithWxJavaTolerance(target, effectiveFieldValues, loader);
        applyFlowSemanticHintsWithWxJavaTolerance(target, effectiveFieldValues, loader);
        if (shouldApplyCommonsValidatorFixups(target, effectiveFieldValues)) {
            applyCommonsValidatorFixups(target);
        }
        applyNinjaCoreSemanticDefaults(target, effectiveFieldValues, loader);
        applyWxJavaSemanticDefaults(target, effectiveFieldValues, loader);
        applyKafkaKeyvalueSemanticDefaults(target, effectiveFieldValues, loader);
    }

    private boolean shouldApplyCommonsValidatorFixups(Object target, Map<String, Object> fieldValues) {
        if (target != null) {
            String className = target.getClass().getName();
            if (className.startsWith("org.apache.commons.validator.")
                    || className.contains("ValidatorResults")
                    || className.contains("ValidatorAction")) {
                return true;
            }
            if (Proxy.isProxyClass(target.getClass())) {
                return false;
            }
        }
        if (fieldValues == null || fieldValues.isEmpty()) {
            return false;
        }
        for (String key : fieldValues.keySet()) {
            if (key != null && key.contains("org.apache.commons.validator")) {
                return true;
            }
        }
        return false;
    }

    private void applyFlowSemanticHints(Object target,
                                        Map<String, Object> fieldValues,
                                        ClassLoader loader) throws Exception {
        if (target == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }
        applyFlowSemanticHints(target,
                fieldValues,
                fieldValues,
                loader,
                Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    private void applyFlowSemanticHints(Object target,
                                        Map<String, Object> localFieldValues,
                                        Map<String, Object> rootFieldValues,
                                        ClassLoader loader,
                                        Set<Object> visited) throws Exception {
        if (target == null || visited.contains(target)) {
            return;
        }
        visited.add(target);

        Map<String, Object> effectiveHints = new LinkedHashMap<>();
        if (rootFieldValues != null) {
            effectiveHints.putAll(collectFlowTypeAliasHints(rootFieldValues, target.getClass()));
        }
        if (localFieldValues != null) {
            effectiveHints.putAll(localFieldValues);
        }

        if (!effectiveHints.isEmpty()) {
            Map<String, Object> previousContext = currentFieldValueContext;
            currentFieldValueContext = rootFieldValues == null ? effectiveHints : rootFieldValues;
            try {
                applyFieldValues(target, effectiveHints, loader);
                applyMethodLikeHints(target, effectiveHints, loader);
            } finally {
                currentFieldValueContext = previousContext == null ? Collections.emptyMap() : previousContext;
            }
        }

        maybeInstallFlowSuperDelegate(target, effectiveHints, loader);
        bridgeFlowDelegateAliases(target, rootFieldValues, loader);

        if (target instanceof Map<?, ?>) {
            for (Object value : ((Map<?, ?>) target).values()) {
                if (value == null) {
                    continue;
                }
                Map<String, Object> valueHints = collectFlowTypeAliasHints(rootFieldValues, value.getClass());
                applyFlowSemanticHints(value, valueHints, rootFieldValues, loader, visited);
            }
            return;
        }

        if (target instanceof Collection<?>) {
            for (Object value : (Collection<?>) target) {
                if (value == null) {
                    continue;
                }
                Map<String, Object> valueHints = collectFlowTypeAliasHints(rootFieldValues, value.getClass());
                applyFlowSemanticHints(value, valueHints, rootFieldValues, loader, visited);
            }
            return;
        }

        if (target.getClass().isArray()) {
            int len = Array.getLength(target);
            for (int i = 0; i < len; i++) {
                Object value = Array.get(target, i);
                if (value == null) {
                    continue;
                }
                Map<String, Object> valueHints = collectFlowTypeAliasHints(rootFieldValues, value.getClass());
                applyFlowSemanticHints(value, valueHints, rootFieldValues, loader, visited);
            }
            return;
        }

        for (Field field : getAllInstanceFields(target.getClass())) {
            field.setAccessible(true);
            Map<String, Object> nestedHints = new LinkedHashMap<>();
            nestedHints.putAll(extractRelativeFieldValues(effectiveHints, field.getName()));

            Object nested = null;
            try {
                nested = field.get(target);
            } catch (Throwable ignored) {
            }

            if (nested != null) {
                nestedHints.putAll(collectFlowTypeAliasHints(rootFieldValues, nested.getClass()));
                if (!field.getType().equals(nested.getClass())) {
                    nestedHints.putAll(collectFlowTypeAliasHints(rootFieldValues, field.getType()));
                }
            }

            if (nested == null && !nestedHints.isEmpty()) {
                try {
                    nested = instantiateSpecialAbstractType(field.getType(), nestedHints, loader);
                    if (nested != null) {
                        field.set(target, nested);
                    }
                } catch (Throwable ignored) {
                }
            }

            if (nested != null
                    && (!nestedHints.isEmpty()
                    || nested instanceof Collection<?>
                    || nested instanceof Map<?, ?>
                    || nested.getClass().isArray())) {
                applyFlowSemanticHints(nested, nestedHints, rootFieldValues, loader, visited);
            }
        }
    }

    private Map<String, Object> collectFlowTypeAliasHints(Map<String, Object> fieldValues, Class<?> type) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (fieldValues == null || type == null) {
            return result;
        }

        String simpleName = type.getSimpleName();
        if (simpleName != null && !simpleName.isEmpty()) {
            result.putAll(extractRelativeFieldValues(fieldValues, simpleName));
        }

        String typeName = type.getName();
        if (typeName != null && !typeName.isEmpty()) {
            result.putAll(extractRelativeFieldValues(fieldValues, typeName));
        }

        return result;
    }

    private void maybeInstallFlowSuperDelegate(Object target,
                                               Map<String, Object> fieldValues,
                                               ClassLoader loader) {
        if (target == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }

        Field delegateField = tryFindField(target.getClass(), "delegate");
        if (delegateField == null) {
            return;
        }

        boolean hasSuperHints = fieldValues.keySet().stream()
                .map(this::normalizeFlowHintKey)
                .anyMatch(k -> k.startsWith("super") || k.contains("superwritefile") || k.contains("superappendfile") || k.contains("superreadfile"));

        try {
            delegateField.setAccessible(true);
            Object current = delegateField.get(target);
            boolean isFlowFilerDelegate = "com.lithium.flow.filer.Filer".equals(delegateField.getType().getName())
                    && target.getClass().getName().startsWith("com.lithium.flow.filer.");
            if (!hasSuperHints && !isFlowFilerDelegate) {
                return;
            }
            if (current == null || Proxy.isProxyClass(current.getClass())) {
                Object stub = instantiateSpecialAbstractType(delegateField.getType(), fieldValues, loader);
                if (stub != null) {
                    delegateField.set(target, stub);
                }
            }
        } catch (Throwable ignored) {
        }
    }

    private void bridgeFlowDelegateAliases(Object target,
                                           Map<String, Object> fieldValues,
                                           ClassLoader loader) throws Exception {
        if (target == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }

        Field hdfsField = tryFindField(target.getClass(), "hdfsFiler");
        if (hdfsField == null) {
            return;
        }

        hdfsField.setAccessible(true);
        Object hdfsFiler = hdfsField.get(target);
        if (hdfsFiler == null) {
            return;
        }

        Field delegateField = tryFindField(hdfsFiler.getClass(), "delegate");
        if (delegateField == null) {
            return;
        }

        delegateField.setAccessible(true);
        Object existingDelegate = delegateField.get(hdfsFiler);
        if (existingDelegate == null
                || Proxy.isProxyClass(existingDelegate.getClass())) {
            applyRelativePrefixes(hdfsFiler, fieldValues, loader, "delegate");
            maybeInstallFlowSuperDelegate(hdfsFiler, fieldValues, loader);
        }

        Object delegate = delegateField.get(hdfsFiler);
        if (delegate == null) {
            return;
        }

        applyRelativePrefixes(delegate, fieldValues, loader, "streamer", "super");
        maybeInstallFlowSuperDelegate(delegate, fieldValues, loader);

        Field streamerField = tryFindField(delegate.getClass(), "streamer");
        if (streamerField == null) {
            return;
        }

        streamerField.setAccessible(true);
        Object streamer = streamerField.get(delegate);
        if (streamer == null) {
            return;
        }

        applyRelativePrefixes(streamer, fieldValues, loader, "outCoder", "compressOption", "inCoder");
    }

    private void applyCommonsValidatorSemanticHints(MethodDescriptor entryDescriptor,
                                                    Object target,
                                                    Object[] args,
                                                    Map<String, Object> fieldValues,
                                                    ClassLoader loader) throws Exception {
        if (entryDescriptor == null || args == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }
        if (!entryDescriptor.getClassName().startsWith("org.apache.commons.validator.")) {
            return;
        }

        Map<String, Object> params = null;
        Map<String, Object> actions = null;
        Object validatorAction = null;
        Object results = null;
        Object field = null;

        String owner = entryDescriptor.getClassName();
        String method = entryDescriptor.getMethodName();

        if ("org.apache.commons.validator.ValidatorAction".equals(owner)
                && "executeValidationMethod".equals(method)
                && args.length >= 4) {
            field = args[0];
            params = castStringObjectMap(args[1]);
            results = args[2];
            validatorAction = target;
            applyRelativePrefixes(validatorAction, fieldValues, loader, "va", "validatorAction", "ValidatorAction");
            applyRelativePrefixes(field, fieldValues, loader, "field", "Field");
            applyRelativePrefixes(results, fieldValues, loader, "results", "ValidatorResults");
            applyRelativePrefixes(params, fieldValues, loader, "params");
        } else if ("org.apache.commons.validator.Field".equals(owner)
                && ("validateForRule".equals(method) || "runDependentValidators".equals(method))
                && args.length >= 5) {
            validatorAction = args[0];
            results = args[1];
            actions = castStringObjectMap(args[2]);
            params = castStringObjectMap(args[3]);
            field = target;
            applyRelativePrefixes(validatorAction, fieldValues, loader, "va", "validatorAction", "ValidatorAction");
            applyRelativePrefixes(results, fieldValues, loader, "results", "ValidatorResults");
            applyRelativePrefixes(actions, fieldValues, loader, "actions");
            applyRelativePrefixes(params, fieldValues, loader, "params");
            applyRelativePrefixes(field, fieldValues, loader, "field", "Field", "org.apache.commons.validator.Field");
        }

        if (params != null) {
            seedCommonsValidatorParams(params, field, validatorAction, results, fieldValues, loader);
        }
        if (actions != null) {
            seedCommonsValidatorActions(actions, fieldValues, loader);
        }

        if (validatorAction != null) {
            applyCommonsValidatorFixups(validatorAction);
        }
        if (field != null) {
            applyCommonsValidatorFixups(field);
        }
    }

    private void applyRelativePrefixes(Object target,
                                       Map<String, Object> fieldValues,
                                       ClassLoader loader,
                                       String... prefixes) throws Exception {
        if (target == null || fieldValues == null || fieldValues.isEmpty() || prefixes == null) {
            return;
        }

        Map<String, Object> merged = new LinkedHashMap<>();
        for (String prefix : prefixes) {
            if (prefix == null || prefix.isEmpty()) {
                continue;
            }
            merged.putAll(extractRelativeFieldValues(fieldValues, prefix));
        }

        if (!merged.isEmpty()) {
            applyHandleAwareFieldValues(target, merged, loader);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> castStringObjectMap(Object value) {
        if (value instanceof Map<?, ?>) {
            return (Map<String, Object>) value;
        }
        return null;
    }

    private void seedCommonsValidatorParams(Map<String, Object> params,
                                            Object field,
                                            Object validatorAction,
                                            Object results,
                                            Map<String, Object> fieldValues,
                                            ClassLoader loader) throws Exception {
        if (params == null) {
            return;
        }

        if (results != null) {
            params.putIfAbsent("org.apache.commons.validator.ValidatorResults", results);
        }
        if (field != null) {
            params.putIfAbsent("org.apache.commons.validator.Field", field);
        }
        if (validatorAction != null) {
            params.putIfAbsent("org.apache.commons.validator.ValidatorAction", validatorAction);
        }

        Object beanHint = firstNonNull(
                fieldValues.get("paramValues[beanIndex]"),
                fieldValues.get("va.getParameterValues(params)[beanIndex]"),
                fieldValues.get("action(name=depRule1).paramValues[beanIndex]"),
                fieldValues.get("during.executeValidationMethod.paramValues[beanIndex]"),
                fieldValues.get("this.getParameterValues(params)[0]"),
                fieldValues.get("this.getParameterValues(arg1)[0]"),
                fieldValues.get("paramValues[0]"),
                fieldValues.get("arg0[\"bean\"]"),
                fieldValues.get("arg1[\"bean\"]"));
        if (beanHint != null && !params.containsKey("java.lang.Object")) {
            params.put("java.lang.Object", adaptKnownMapValue("java.lang.Object", beanHint, loader));
        }
    }

    private void seedCommonsValidatorActions(Map<String, Object> actions,
                                             Map<String, Object> fieldValues,
                                             ClassLoader loader) throws Exception {
        if (actions == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }

        Map<String, Object> directEntries = new LinkedHashMap<>();
        Map<String, Map<String, Object>> nestedEntries = new LinkedHashMap<>();

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (key == null) {
                continue;
            }

            String actionName = extractCommonsValidatorActionName(key);
            if (actionName == null) {
                continue;
            }

            int dotPos = key.indexOf("].");
            if (dotPos >= 0 && key.startsWith("actions[")) {
                nestedEntries.computeIfAbsent(actionName, k -> new LinkedHashMap<>())
                        .put(key.substring(dotPos + 2), entry.getValue());
                continue;
            }

            if (key.startsWith("actions.get(") && key.endsWith(")")) {
                directEntries.put(actionName, entry.getValue());
                continue;
            }

            if (key.startsWith("actions[") && key.endsWith("]")) {
                directEntries.put(actionName, entry.getValue());
                continue;
            }

            if (key.startsWith("action(name=") && key.contains(").")) {
                nestedEntries.computeIfAbsent(actionName, k -> new LinkedHashMap<>())
                        .put(key.substring(key.indexOf(").") + 2), entry.getValue());
            }
        }

        for (Map.Entry<String, Object> entry : directEntries.entrySet()) {
            String actionName = entry.getKey();
            Object rawValue = entry.getValue();
            Object current = actions.get(actionName);
            if (current == null && rawValue instanceof String && isSymbolicHandle((String) rawValue)) {
                current = rebuildObjectFromHandle((String) rawValue,
                        resolveKnownMapValueType("org.apache.commons.validator.ValidatorAction", loader),
                        fieldValues,
                        loader);
            }
            if (current == null && rawValue != null) {
                current = adaptKnownMapValue("org.apache.commons.validator.ValidatorAction", rawValue, loader);
            }
            if (current != null) {
                actions.put(actionName, current);
                ensureCommonsValidatorActionName(current, actionName);
            }
        }

        for (Map.Entry<String, Map<String, Object>> entry : nestedEntries.entrySet()) {
            Object action = actions.get(entry.getKey());
            if (action == null) {
                Class<?> actionType = resolveKnownMapValueType("org.apache.commons.validator.ValidatorAction", loader);
                if (actionType != null) {
                    action = instantiateClass(actionType, fieldValues, loader);
                    actions.put(entry.getKey(), action);
                } else {
                    continue;
                }
            }
            if (action == null) {
                continue;
            }
            applyHandleAwareFieldValues(action, entry.getValue(), loader);
            ensureCommonsValidatorActionName(action, entry.getKey());
        }
    }

    private void ensureCommonsValidatorActionName(Object action, String actionName) {
        if (action == null || actionName == null || actionName.isEmpty()) {
            return;
        }
        try {
            Field nameField = tryFindField(action.getClass(), "name");
            if (nameField == null) {
                return;
            }
            nameField.setAccessible(true);
            Object current = nameField.get(action);
            if (!(current instanceof String) || ((String) current).trim().isEmpty()) {
                nameField.set(action, actionName);
            }
        } catch (Throwable ignored) {
        }
    }

    private String extractCommonsValidatorActionName(String key) {
        if (key == null) {
            return null;
        }
        if (key.startsWith("actions[")) {
            int start = key.indexOf('[') + 1;
            int end = key.indexOf(']', start);
            if (start > 0 && end > start) {
                return stripEnclosingQuotes(key.substring(start, end).trim());
            }
        }
        if (key.startsWith("actions.get(")) {
            int start = key.indexOf('(') + 1;
            int end = key.indexOf(')', start);
            if (start > 0 && end > start) {
                return stripEnclosingQuotes(key.substring(start, end).trim());
            }
        }
        if (key.startsWith("action(name=")) {
            int start = "action(name=".length();
            int end = key.indexOf(')', start);
            if (end > start) {
                return stripEnclosingQuotes(key.substring(start, end).trim());
            }
        }
        return null;
    }

    private Object firstNonNull(Object... values) {
        if (values == null) {
            return null;
        }
        for (Object value : values) {
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private void applyCommonsValidatorFixups(Object root) throws Exception {
        applyCommonsValidatorFixups(root, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    private void applyCommonsValidatorFixups(Object root, Set<Object> visited) throws Exception {
        if (root == null || visited.contains(root)) {
            return;
        }
        visited.add(root);

        normalizeCommonsValidatorObject(root);
        if (isCommonsValidatorAction(root)) {
            seedSyntheticValidationMethod(root);
        }

        if (root instanceof Map<?, ?>) {
            for (Object value : ((Map<?, ?>) root).values()) {
                applyCommonsValidatorFixups(value, visited);
            }
            return;
        }

        if (root instanceof Collection<?>) {
            for (Object value : (Collection<?>) root) {
                applyCommonsValidatorFixups(value, visited);
            }
            return;
        }

        if (root.getClass().isArray()) {
            int len = Array.getLength(root);
            for (int i = 0; i < len; i++) {
                applyCommonsValidatorFixups(Array.get(root, i), visited);
            }
            return;
        }

        for (Field field : getAllInstanceFields(root.getClass())) {
            field.setAccessible(true);
            try {
                applyCommonsValidatorFixups(field.get(root), visited);
            } catch (Throwable ignored) {
            }
        }
    }

    private void normalizeCommonsValidatorObject(Object root) {
        if (root == null) {
            return;
        }
        String className = root.getClass().getName();
        if ("org.apache.commons.validator.Field".equals(className)) {
            normalizeCommonsValidatorField(root);
        } else if ("org.apache.commons.validator.ValidatorAction".equals(className)) {
            normalizeCommonsValidatorAction(root);
        }
    }

    @SuppressWarnings("unchecked")
    private void normalizeCommonsValidatorField(Object fieldObj) {
        try {
            Field dependsField = tryFindField(fieldObj.getClass(), "depends");
            Field dependencyListField = tryFindField(fieldObj.getClass(), "dependencyList");
            Field keyField = tryFindField(fieldObj.getClass(), "key");
            Field propertyField = tryFindField(fieldObj.getClass(), "property");
            Field indexedListPropertyField = tryFindField(fieldObj.getClass(), "indexedListProperty");
            if (dependsField != null && dependencyListField != null) {
                dependsField.setAccessible(true);
                dependencyListField.setAccessible(true);
                Object rawDepends = dependsField.get(fieldObj);
                if (rawDepends instanceof String && !((String) rawDepends).trim().isEmpty()) {
                    List<String> normalized = tokenizeCsv((String) rawDepends);
                    List<String> target = (List<String>) dependencyListField.get(fieldObj);
                    if (target != null) {
                        target.clear();
                        target.addAll(normalized);
                    }
                }
            }

            if (keyField != null) {
                keyField.setAccessible(true);
                Object currentKey = keyField.get(fieldObj);
                if (!(currentKey instanceof String) || ((String) currentKey).trim().isEmpty()) {
                    String property = readStringField(fieldObj, propertyField);
                    String indexedListProperty = readStringField(fieldObj, indexedListPropertyField);
                    if (property != null && !property.isEmpty()) {
                        keyField.set(fieldObj,
                                indexedListProperty != null && !indexedListProperty.isEmpty()
                                        ? indexedListProperty + "[]" + "." + property
                                        : property);
                    }
                }
            }
        } catch (Throwable ignored) {
        }
    }

    @SuppressWarnings("unchecked")
    private void normalizeCommonsValidatorAction(Object actionObj) {
        try {
            Field dependsField = tryFindField(actionObj.getClass(), "depends");
            Field dependencyListField = tryFindField(actionObj.getClass(), "dependencyList");
            if (dependsField != null && dependencyListField != null) {
                dependsField.setAccessible(true);
                dependencyListField.setAccessible(true);
                Object rawDepends = dependsField.get(actionObj);
                if (rawDepends instanceof String && !((String) rawDepends).trim().isEmpty()) {
                    List<String> normalized = tokenizeCsv((String) rawDepends);
                    List<String> target = (List<String>) dependencyListField.get(actionObj);
                    if (target != null) {
                        target.clear();
                        target.addAll(normalized);
                    }
                }
            }

            Field methodParamsField = tryFindField(actionObj.getClass(), "methodParams");
            Field methodParameterListField = tryFindField(actionObj.getClass(), "methodParameterList");
            if (methodParamsField != null && methodParameterListField != null) {
                methodParamsField.setAccessible(true);
                methodParameterListField.setAccessible(true);
                Object rawMethodParams = methodParamsField.get(actionObj);
                if (rawMethodParams instanceof String && !((String) rawMethodParams).trim().isEmpty()) {
                    List<String> normalized = tokenizeCsv((String) rawMethodParams);
                    List<String> target = (List<String>) methodParameterListField.get(actionObj);
                    if (target != null) {
                        target.clear();
                        target.addAll(normalized);
                    }
                }
            }
        } catch (Throwable ignored) {
        }
    }

    private String readStringField(Object owner, Field field) throws IllegalAccessException {
        if (owner == null || field == null) {
            return null;
        }
        field.setAccessible(true);
        Object value = field.get(owner);
        return value instanceof String ? ((String) value).trim() : null;
    }

    private List<String> tokenizeCsv(String csv) {
        List<String> result = new ArrayList<>();
        if (csv == null) {
            return result;
        }
        StringTokenizer tokenizer = new StringTokenizer(csv, ",");
        while (tokenizer.hasMoreTokens()) {
            String token = tokenizer.nextToken().trim();
            if (!token.isEmpty()) {
                result.add(token);
            }
        }
        return result;
    }

    private boolean isCommonsValidatorAction(Object value) {
        return value != null && "org.apache.commons.validator.ValidatorAction".equals(value.getClass().getName());
    }

    private List<Field> getAllInstanceFields(Class<?> type) {
        List<Field> fields = new ArrayList<>();
        Class<?> current = type;
        while (current != null && current != Object.class) {
            for (Field field : current.getDeclaredFields()) {
                if (!Modifier.isStatic(field.getModifiers())) {
                    fields.add(field);
                }
            }
            current = current.getSuperclass();
        }
        return fields;
    }

    private void seedSyntheticValidationMethod(Object validatorAction) throws Exception {
        Field validationMethodField = tryFindField(validatorAction.getClass(), "validationMethod");
        Field methodParameterListField = tryFindField(validatorAction.getClass(), "methodParameterList");
        Field dependencyListField = tryFindField(validatorAction.getClass(), "dependencyList");
        if (validationMethodField == null || methodParameterListField == null) {
            return;
        }

        validationMethodField.setAccessible(true);
        methodParameterListField.setAccessible(true);
        if (dependencyListField != null) {
            dependencyListField.setAccessible(true);
            if (dependencyListField.get(validatorAction) == null) {
                dependencyListField.set(validatorAction, new ArrayList<>());
            }
        }

        Object currentValidationMethod = validationMethodField.get(validatorAction);
        if (isUsableValidationMethod(currentValidationMethod, methodParameterListField.get(validatorAction))) {
            return;
        }

        Object rawParameterList = methodParameterListField.get(validatorAction);
        int size = rawParameterList instanceof Collection<?> ? ((Collection<?>) rawParameterList).size() : 0;
        Method synthetic = syntheticValidatorMethodForArity(size);
        if (synthetic != null) {
            validationMethodField.set(validatorAction, synthetic);
        }
    }

    private boolean isUsableValidationMethod(Object validationMethod, Object rawParameterList) {
        if (!(validationMethod instanceof Method)) {
            return false;
        }
        try {
            Method method = (Method) validationMethod;
            if (method.getDeclaringClass() == null || method.getName() == null) {
                return false;
            }
            int expectedArity = rawParameterList instanceof Collection<?> ? ((Collection<?>) rawParameterList).size() : -1;
            return expectedArity < 0 || method.getParameterCount() == expectedArity;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private Method syntheticValidatorMethodForArity(int size) throws NoSuchMethodException {
        if (size <= 0) {
            return null;
        }
        if (size == 1) {
            return PoCReplayRunner.class.getDeclaredMethod("syntheticValidatorMethod1", Object.class);
        }
        if (size == 2) {
            return PoCReplayRunner.class.getDeclaredMethod("syntheticValidatorMethod2", Object.class, Object.class);
        }
        if (size == 3) {
            return PoCReplayRunner.class.getDeclaredMethod("syntheticValidatorMethod3", Object.class, Object.class, Object.class);
        }
        if (size == 4) {
            return PoCReplayRunner.class.getDeclaredMethod("syntheticValidatorMethod4", Object.class, Object.class, Object.class, Object.class);
        }
        return null;
    }

    public static Boolean syntheticValidatorMethod1(Object a0) {
        return Boolean.TRUE;
    }

    public static Boolean syntheticValidatorMethod2(Object a0, Object a1) {
        return Boolean.TRUE;
    }

    public static Boolean syntheticValidatorMethod3(Object a0, Object a1, Object a2) {
        return Boolean.TRUE;
    }

    public static Boolean syntheticValidatorMethod4(Object a0, Object a1, Object a2, Object a3) {
        return Boolean.TRUE;
    }

    private Map<String, Object> extractRelativeFieldValues(Map<String, Object> fieldValues, String prefix) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (fieldValues == null || prefix == null || prefix.isEmpty()) {
            return result;
        }

        String dotPrefix = prefix + ".";
        String bracketPrefix = prefix + "[";
        String underscorePrefix = prefix + "_";

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String originalKey = entry.getKey();
            if (originalKey == null) {
                continue;
            }

            for (String key : candidateSyntheticKeys(originalKey)) {
                if (key.startsWith(dotPrefix)) {
                    result.put(key.substring(dotPrefix.length()), entry.getValue());
                    break;
                }

                if (key.startsWith(bracketPrefix)) {
                    result.put(key.substring(prefix.length()), entry.getValue());
                    break;
                }

                if (key.startsWith(underscorePrefix)) {
                    String normalized = normalizeSyntheticPath(key.substring(prefix.length()));
                    if (!normalized.isEmpty()) {
                        result.put(normalized, entry.getValue());
                        break;
                    }
                }
            }
        }

        return result;
    }

    private List<String> candidateSyntheticKeys(String rawKey) {
        if (rawKey == null || rawKey.isEmpty()) {
            return Collections.emptyList();
        }

        LinkedHashSet<String> candidates = new LinkedHashSet<>();
        candidates.add(rawKey);

        String castStripped = stripSyntheticCasts(rawKey);
        if (!castStripped.equals(rawKey)) {
            candidates.add(castStripped);
        }

        return new ArrayList<>(candidates);
    }

    private String stripSyntheticCasts(String rawKey) {
        if (rawKey == null || rawKey.isEmpty()) {
            return "";
        }

        String current = rawKey;
        boolean changed = false;
        while (true) {
            String next = current
                    .replaceAll("\\(\\s*[A-Za-z_$][\\w$]*(?:\\.[A-Za-z_$][\\w$]*)+\\s*\\)", "")
                    .replaceAll("\\(([^()\"']+)\\)", "$1");
            if (next.equals(current)) {
                return changed ? next : rawKey;
            }
            current = next;
            changed = true;
        }
    }

    private Map<String, Object> collectArgumentFieldValues(String argKey,
                                                           Object rawArgValue,
                                                           Class<?> parameterType,
                                                           Map<String, Object> fieldValues) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (fieldValues == null || fieldValues.isEmpty()) {
            return result;
        }

        for (String alias : deriveArgumentAliases(parameterType)) {
            result.putAll(extractRelativeFieldValues(fieldValues, alias));
        }
        if (rawArgValue instanceof String && isSymbolicHandle((String) rawArgValue)) {
            result.putAll(expandHandleScopedFieldValues((String) rawArgValue, fieldValues, new LinkedHashSet<>()));
        }
        result.putAll(extractRelativeFieldValues(fieldValues, argKey));
        return result;
    }

    private List<String> deriveArgumentAliases(Class<?> parameterType) {
        if (parameterType == null) {
            return Collections.emptyList();
        }

        LinkedHashSet<String> aliases = new LinkedHashSet<>();
        String simpleName = parameterType.getSimpleName();
        if (simpleName != null && !simpleName.isEmpty()) {
            aliases.add(simpleName);
            aliases.add(Introspector.decapitalize(simpleName));
            if ("ClientConfig".equals(simpleName)) {
                aliases.add("config");
            } else if ("Request".equals(simpleName)) {
                aliases.add("request");
            } else if ("Executor".equals(simpleName)) {
                aliases.add("executor");
            }
        }
        return new ArrayList<>(aliases);
    }

    private Map<String, Object> expandHandleScopedFieldValues(String handle,
                                                              Map<String, Object> fieldValues,
                                                              Set<String> visiting) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (handle == null || fieldValues == null || fieldValues.isEmpty() || !visiting.add(handle)) {
            return result;
        }

        try {
            for (Map.Entry<String, Object> entry : extractHandleRelativeFieldValues(fieldValues, handle).entrySet()) {
                String normalizedPath = normalizeSyntheticPath(entry.getKey());
                if (normalizedPath.isEmpty()) {
                    continue;
                }

                Object value = entry.getValue();
                result.put(normalizedPath, value);

                if (value instanceof String && isSymbolicHandle((String) value)) {
                    Map<String, Object> nested = expandHandleScopedFieldValues((String) value, fieldValues, visiting);
                    for (Map.Entry<String, Object> nestedEntry : nested.entrySet()) {
                        result.put(normalizedPath + "." + nestedEntry.getKey(), nestedEntry.getValue());
                    }
                }
            }
        } finally {
            visiting.remove(handle);
        }
        return result;
    }

    private Map<String, Object> extractHandleRelativeFieldValues(Map<String, Object> fieldValues, String handle) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (fieldValues == null || handle == null || handle.isEmpty()) {
            return result;
        }

        String hashPrefix = handle + "#";
        String dotPrefix = handle + ".";

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (key == null) {
                continue;
            }
            if (key.startsWith(hashPrefix)) {
                result.put(key.substring(hashPrefix.length()), entry.getValue());
                continue;
            }
            if (key.startsWith(dotPrefix)) {
                result.put(key.substring(dotPrefix.length()), entry.getValue());
            }
        }
        return result;
    }

    private String normalizeSyntheticPath(String rawPath) {
        if (rawPath == null) {
            return "";
        }

        String path = rawPath.trim().replace('#', '.');
        while (path.startsWith(".")) {
            path = path.substring(1);
        }
        if (path.isEmpty()) {
            return "";
        }

        List<String> tokens = splitSyntheticPathTokens(path);
        List<String> normalized = new ArrayList<>();
        for (String token : tokens) {
            String value = normalizeSyntheticToken(token);
            if (!value.isEmpty()) {
                normalized.add(value);
            }
        }
        return String.join(".", normalized);
    }

    private List<String> splitSyntheticPathTokens(String path) {
        List<String> tokens = new ArrayList<>();
        if (path == null || path.isEmpty()) {
            return tokens;
        }

        StringBuilder current = new StringBuilder();
        int bracketDepth = 0;
        for (int i = 0; i < path.length(); i++) {
            char ch = path.charAt(i);
            if (ch == '.' && bracketDepth == 0) {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
                continue;
            }
            if (ch == '[') {
                bracketDepth++;
            } else if (ch == ']') {
                bracketDepth = Math.max(0, bracketDepth - 1);
            }
            current.append(ch);
        }
        if (current.length() > 0) {
            tokens.add(current.toString());
        }
        return tokens;
    }

    private String normalizeSyntheticToken(String token) {
        if (token == null) {
            return "";
        }

        String t = token.trim();
        if (t.isEmpty()) {
            return "";
        }
        if (t.startsWith("[") && t.endsWith("]")) {
            return normalizePathToken(t);
        }

        t = t.replace("()", "");
        t = t.replace('-', '_');
        t = stripKnownSyntheticSuffix(t);
        if (t.isEmpty()) {
            return "";
        }

        String canonical = canonicalizeWellKnownLiteral(t);
        if (canonical.contains("_")) {
            String[] parts = canonical.split("_+");
            StringBuilder joined = new StringBuilder();
            for (int i = 0; i < parts.length; i++) {
                String part = parts[i];
                if (part.isEmpty()) {
                    continue;
                }
                if (joined.length() == 0) {
                    joined.append(part);
                } else {
                    joined.append(Character.toUpperCase(part.charAt(0)));
                    if (part.length() > 1) {
                        joined.append(part.substring(1));
                    }
                }
            }
            canonical = joined.toString();
        }
        return canonical;
    }

    private String stripKnownSyntheticSuffix(String rawToken) {
        String token = rawToken;
        String[] suffixes = {
                "ObjectIdentity", "objectIdentity",
                "Identity", "identity",
                "Reference", "reference",
                "Instance", "instance",
                "ConcreteType", "concreteType",
                "UriRef", "uriRef",
                "Ref", "ref"
        };
        boolean changed;
        do {
            changed = false;
            for (String suffix : suffixes) {
                if (token.length() > suffix.length() && token.endsWith(suffix)) {
                    token = token.substring(0, token.length() - suffix.length());
                    changed = true;
                    break;
                }
            }
        } while (changed);
        return token;
    }

    private boolean isArgumentScopedKey(String key) {
        return key != null && key.matches("^arg\\d+(\\.|\\[).*$");
    }

    private boolean looksLikeStaticClassPath(String key) {
        if (key == null || key.startsWith("this.") || isArgumentScopedKey(key)) {
            return false;
        }
        int dot = key.indexOf('.');
        if (dot <= 0) {
            return false;
        }
        String first = key.substring(0, dot);
        return !first.isEmpty() && Character.isUpperCase(first.charAt(0));
    }

    private boolean looksLikeFlowInstanceAliasPath(String key) {
        if (key == null || key.isEmpty()) {
            return false;
        }
        if (key.startsWith("com.lithium.flow.")) {
            return true;
        }
        int dot = key.indexOf('.');
        if (dot <= 0) {
            return false;
        }
        String first = key.substring(0, dot);
        int at = first.indexOf('@');
        if (at > 0) {
            first = first.substring(0, at);
        }
        return !first.isEmpty() && Character.isUpperCase(first.charAt(0));
    }

    private Object buildMissingReferenceArg(int argIndex,
                                            Class<?> targetType,
                                            ClassLoader loader) throws Exception {
        log.info("[DEBUG buildMissingReferenceArg] argIndex={}, targetType={}",
                argIndex, targetType.getName());

        if (targetType == Object.class) {
            return new Object();
        }

        if (targetType == String.class) {
            return "";
        }

        if (Map.class.isAssignableFrom(targetType)) {
            return new LinkedHashMap<>();
        }

        if (Collection.class.isAssignableFrom(targetType)) {
            return new ArrayList<>();
        }

        if (targetType.isArray()) {
            return Array.newInstance(targetType.getComponentType(), 0);
        }

        Object special = instantiateSpecialAbstractType(targetType, Collections.emptyMap(), loader);
        if (special != null) {
            return special;
        }

        if (!targetType.isInterface() && !Modifier.isAbstract(targetType.getModifiers())) {
            try {
                return instantiateClass(targetType, Collections.emptyMap(), loader);
            } catch (Exception e) {
                log.info("[DEBUG buildMissingReferenceArg] instantiateClass failed for {}: {}",
                        targetType.getName(), e.toString());
            }
        }

        return new Object();
    }

    /**
     * Generic object-graph recovery:
     * 1. Supports this.xxx.yyy paths.
     * 2. Automatically creates intermediate objects.
     * 3. Routes unmatched fields into child objects held by a Collection when possible.
     */
    private void applyFieldValues(Object target, Map<String, Object> fieldValues, ClassLoader loader) throws Exception {
        if (target == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }

        Map<String, Object> previousContext = currentFieldValueContext;
        currentFieldValueContext = fieldValues;
        try {
            AssignmentNode root = buildAssignmentTree(fieldValues);
            applyAssignmentNode(target, root, loader, "(root)");
        } finally {
            currentFieldValueContext = previousContext == null ? Collections.emptyMap() : previousContext;
        }
    }

    private void applyFieldValuesWithWxJavaTolerance(Object target,
                                                     Map<String, Object> fieldValues,
                                                     ClassLoader loader) throws Exception {
        try {
            applyFieldValues(target, fieldValues, loader);
        } catch (IllegalArgumentException e) {
            if (isWxJavaReplayObject(target) && looksLikeWxJavaSemanticConfigError(e)) {
                log.info("[DEBUG applyFieldValuesWithWxJavaTolerance] ignored WxJava semantic config hint for {}: {}",
                        target.getClass().getName(), e.toString());
                return;
            }
            throw e;
        }
    }

    private void applyMethodLikeHintsWithWxJavaTolerance(Object target,
                                                         Map<String, Object> fieldValues,
                                                         ClassLoader loader) throws Exception {
        try {
            applyMethodLikeHints(target, fieldValues, loader);
        } catch (IllegalArgumentException e) {
            if (isWxJavaReplayObject(target) && looksLikeWxJavaSemanticConfigError(e)) {
                log.info("[DEBUG applyMethodLikeHintsWithWxJavaTolerance] ignored WxJava semantic method hint for {}: {}",
                        target.getClass().getName(), e.toString());
                return;
            }
            throw e;
        }
    }

    private void applyFlowSemanticHintsWithWxJavaTolerance(Object target,
                                                           Map<String, Object> fieldValues,
                                                           ClassLoader loader) throws Exception {
        try {
            applyFlowSemanticHints(target, fieldValues, loader);
        } catch (IllegalArgumentException e) {
            if (isWxJavaReplayObject(target) && looksLikeWxJavaSemanticConfigError(e)) {
                log.info("[DEBUG applyFlowSemanticHintsWithWxJavaTolerance] ignored WxJava semantic flow hint for {}: {}",
                        target.getClass().getName(), e.toString());
                return;
            }
            throw e;
        }
    }

    private boolean isWxJavaReplayObject(Object target) {
        return target != null && target.getClass().getName().startsWith("me.chanjar.weixin.");
    }

    private boolean looksLikeWxJavaSemanticConfigError(Throwable e) {
        String message = e == null ? "" : String.valueOf(e);
        return message.contains("WxMpConfigStorage") || message.contains("configStorageMap");
    }

    private AssignmentNode buildAssignmentTree(Map<String, Object> fieldValues) {
        AssignmentNode root = new AssignmentNode();

        List<String> keys = new ArrayList<>(fieldValues.keySet());
        keys.sort(Comparator.comparingInt(this::pathDepth));

        for (String rawKey : keys) {
            String key = normalizePath(rawKey);
            if (key.isEmpty()) {
                continue;
            }
            insertAssignment(root, key, fieldValues.get(rawKey));
        }
        return root;
    }

    private int pathDepth(String rawKey) {
        String key = normalizePath(rawKey);
        if (key.isEmpty()) {
            return 0;
        }
        return splitPathTokens(key).size();
    }

    private String normalizePath(String rawKey) {
        if (rawKey == null) {
            return "";
        }
        String key = rawKey.trim();
        if (key.startsWith("this.")) {
            key = key.substring("this.".length());
        } else if (key.equals("this")) {
            key = "";
        }
        while (key.startsWith(".")) {
            key = key.substring(1);
        }
        return key;
    }

    private void insertAssignment(AssignmentNode root, String path, Object value) {
        AssignmentNode current = root;
        for (String part : splitPathTokens(path)) {
            current = current.children.computeIfAbsent(part, k -> new AssignmentNode());
        }
        current.value = value;
    }

    private List<String> splitPathTokens(String path) {
        List<String> tokens = new ArrayList<>();
        if (path == null || path.isEmpty()) {
            return tokens;
        }

        StringBuilder current = new StringBuilder();
        int bracketDepth = 0;

        for (int i = 0; i < path.length(); i++) {
            char ch = path.charAt(i);

            if (ch == '.' && bracketDepth == 0) {
                flushPathToken(tokens, current);
                continue;
            }

            if (ch == '[') {
                if (bracketDepth == 0 && current.length() > 0) {
                    flushPathToken(tokens, current);
                }
                current.append(ch);
                bracketDepth++;
                continue;
            }

            current.append(ch);
            if (ch == ']') {
                bracketDepth = Math.max(0, bracketDepth - 1);
                if (bracketDepth == 0) {
                    flushPathToken(tokens, current);
                }
            }
        }

        flushPathToken(tokens, current);
        return tokens;
    }

    private void flushPathToken(List<String> tokens, StringBuilder current) {
        if (current.length() == 0) {
            return;
        }
        String rawToken = current.toString().trim();
        current.setLength(0);
        if (rawToken.isEmpty()) {
            return;
        }
        tokens.add(normalizePathToken(rawToken));
    }

    private String normalizePathToken(String token) {
        if (token == null) {
            return "";
        }

        String t = token.trim();
        if (t.startsWith("[") && t.endsWith("]")) {
            String inside = stripEnclosingQuotes(t.substring(1, t.length() - 1).trim());
            return "[" + canonicalizeWellKnownLiteral(inside) + "]";
        }

        if (t.endsWith("()")) {
            if (t.startsWith("get") && t.length() > 5) {
                return Introspector.decapitalize(t.substring(3, t.length() - 2));
            }
            if (t.startsWith("is") && t.length() > 4) {
                return Introspector.decapitalize(t.substring(2, t.length() - 2));
            }
        }

        if (t.startsWith("get(") && t.endsWith(")") && t.length() > 5) {
            String inside = stripEnclosingQuotes(t.substring(4, t.length() - 1).trim());
            return "[" + canonicalizeWellKnownLiteral(inside) + "]";
        }

        return canonicalizeWellKnownLiteral(t);
    }

    private String stripEnclosingQuotes(String raw) {
        if (raw == null || raw.length() < 2) {
            return raw;
        }
        char first = raw.charAt(0);
        char last = raw.charAt(raw.length() - 1);
        if ((first == '\'' && last == '\'') || (first == '"' && last == '"')) {
            return raw.substring(1, raw.length() - 1).trim();
        }
        return raw;
    }

    private void applyAssignmentNode(Object obj,
                                     AssignmentNode node,
                                     ClassLoader loader,
                                     String objectPath) throws Exception {
        if (obj == null || node == null) {
            return;
        }

        Map<String, AssignmentNode> unmatched = new LinkedHashMap<>();

        for (Map.Entry<String, AssignmentNode> entry : node.children.entrySet()) {
            String childName = entry.getKey();
            AssignmentNode childNode = entry.getValue();

            if (isBracketToken(childName)) {
                if (applyBracketAssignment(obj, childName, childNode, loader, objectPath)) {
                    continue;
                }
                unmatched.put(childName, childNode);
                continue;
            }

            if (obj instanceof Map<?, ?>) {
                CollapsedMapKey collapsed = collapseMapKeyChain(childName, childNode);
                String effectiveName = collapsed == null ? childName : collapsed.key;
                AssignmentNode effectiveNode = collapsed == null ? childNode : collapsed.node;
                if (applyMapChildAssignment(obj, effectiveName, effectiveNode, loader, objectPath)) {
                    continue;
                }
                unmatched.put(effectiveName, effectiveNode);
                continue;
            }

            Field field = tryFindField(obj.getClass(), childName);
            if (field == null) {
                unmatched.put(childName, childNode);
                continue;
            }

            field.setAccessible(true);

            if (childNode.children.isEmpty()) {
                if (Collection.class.isAssignableFrom(field.getType())) {
                    try {
                        Object rebuiltCollection = instantiateCollectionFieldObject(field, childNode, loader);
                        if (rebuiltCollection != null) {
                            field.set(obj, rebuiltCollection);
                            log.info("[DEBUG applyFieldValues] set collection field {}.{} = {} (class={})",
                                    objectPath,
                                    field.getName(),
                                    rebuiltCollection,
                                    rebuiltCollection.getClass().getName());
                            continue;
                        }
                    } catch (Exception e) {
                        log.warn("[DEBUG applyFieldValues] failed to set collection field {}.{} with value {}: {}",
                                objectPath, field.getName(), childNode.value, e.toString());
                    }
                }

                // Leaf node: assign the value directly to the current field.
                try {
                    Object convertedValue = convertValueOrTreatAsHint(childNode.value, field.getType(), loader);
                    field.set(obj, convertedValue);
                    log.info("[DEBUG applyFieldValues] set {}.{} = {} (class={})",
                            objectPath,
                            field.getName(),
                            convertedValue,
                            convertedValue == null ? "null" : convertedValue.getClass().getName());
                } catch (Exception e) {
                    log.warn("[DEBUG applyFieldValues] failed to set leaf field {}.{} with value {}: {}",
                            objectPath, field.getName(), childNode.value, e.toString());
                }
                continue;
            }

            // Non-leaf node: make sure the intermediate object exists first.
            Object currentFieldObj = null;
            try {
                currentFieldObj = field.get(obj);
            } catch (Exception ignored) {
            }

            if (currentFieldObj == null) {
                try {
                    currentFieldObj = instantiateFieldObject(field, childNode, loader);
                    if (currentFieldObj != null) {
                        field.set(obj, currentFieldObj);
                        log.info("[DEBUG applyFieldValues] created intermediate object for {}.{} => {}",
                                objectPath, field.getName(), currentFieldObj.getClass().getName());
                    }
                } catch (Exception e) {
                    log.warn("[DEBUG applyFieldValues] failed to create intermediate object for {}.{}: {}",
                            objectPath, field.getName(), e.toString());
                }
            }

            if (Collection.class.isAssignableFrom(field.getType())
                    && hasIndexedChildren(childNode)) {
                try {
                    Object rebuiltCollection = instantiateCollectionFieldObject(field, childNode, loader);
                    if (rebuiltCollection != null) {
                        field.set(obj, rebuiltCollection);
                        currentFieldObj = rebuiltCollection;
                        log.info("[DEBUG applyFieldValues] rebuilt typed collection for {}.{} => {}",
                                objectPath, field.getName(), currentFieldObj.getClass().getName());
                        continue;
                    }
                } catch (Exception e) {
                    log.warn("[DEBUG applyFieldValues] failed to rebuild typed collection for {}.{}: {}",
                            objectPath, field.getName(), e.toString());
                }
            }

            // If this node also has a value, try assigning the whole object first; on failure, keep recursing into children.
            if (childNode.value != null && currentFieldObj == null) {
                try {
                    Object convertedValue = convertValueOrTreatAsHint(childNode.value, field.getType(), loader);
                    field.set(obj, convertedValue);
                    currentFieldObj = field.get(obj);
                    log.info("[DEBUG applyFieldValues] set whole object field {}.{} = {}",
                            objectPath, field.getName(), convertedValue);
                } catch (Exception e) {
                    log.info("[DEBUG applyFieldValues] treat {}.{} scalar value as hint only: {}",
                            objectPath, field.getName(), childNode.value);
                }
            }

            if (currentFieldObj != null) {
                if (currentFieldObj instanceof Map<?, ?>) {
                    applyTypedMapAssignmentNode(castMap(currentFieldObj),
                            childNode,
                            resolveMapValueUpperBound(field, loader),
                            loader,
                            objectPath + "." + field.getName());
                } else {
                    applyAssignmentNode(currentFieldObj, childNode, loader, objectPath + "." + field.getName());
                }
            } else {
                log.warn("[DEBUG applyFieldValues] still null after creation attempt for {}.{}",
                        objectPath, field.getName());
            }
        }

        if (!unmatched.isEmpty()) {
            boolean routed = routeUnmatchedToCollectionChildren(obj, unmatched, loader, objectPath);
            if (!routed) {
                for (String name : unmatched.keySet()) {
                    log.info("[DEBUG applyFieldValues] unmatched path {}.{} ignored",
                            objectPath, name);
                }
            }
        }
    }

    private boolean hasIndexedChildren(AssignmentNode node) {
        if (node == null || node.children.isEmpty()) {
            return false;
        }
        for (String key : node.children.keySet()) {
            if (isBracketToken(key)) {
                return true;
            }
        }
        return false;
    }

    private Field tryFindField(Class<?> clazz, String fieldName) {
        Field exact = tryFindExactField(clazz, fieldName);
        if (exact != null) {
            return exact;
        }

        String wanted = normalizeFieldLookupToken(fieldName);
        if (wanted.isEmpty()) {
            return null;
        }

        Field best = null;
        int bestScore = Integer.MIN_VALUE;
        for (Field field : getAllFields(clazz)) {
            int score = scoreFieldCandidate(field, wanted);
            if (score > bestScore) {
                best = field;
                bestScore = score;
            }
        }
        return bestScore > 0 ? best : null;
    }

    private Field tryFindExactField(Class<?> clazz, String fieldName) {
        Class<?> current = clazz;
        while (current != null) {
            try {
                return current.getDeclaredField(fieldName);
            } catch (NoSuchFieldException ignored) {
                current = current.getSuperclass();
            }
        }
        return null;
    }

    private String normalizeFieldLookupToken(String raw) {
        if (raw == null) {
            return "";
        }
        return raw.replaceAll("[^A-Za-z0-9]", "").toLowerCase(Locale.ROOT);
    }

    private int scoreFieldCandidate(Field field, String wanted) {
        if (field == null || wanted == null || wanted.isEmpty()) {
            return Integer.MIN_VALUE;
        }

        String candidate = normalizeFieldLookupToken(field.getName());
        if (candidate.isEmpty()) {
            return Integer.MIN_VALUE;
        }
        if (candidate.equals(wanted)) {
            return 1000 + candidate.length();
        }
        if (wanted.endsWith(candidate)) {
            return 800 + candidate.length();
        }
        if (wanted.startsWith(candidate)) {
            return 700 + candidate.length();
        }
        if (wanted.contains(candidate)) {
            return 600 + candidate.length();
        }
        if (candidate.contains(wanted)) {
            return 500 + wanted.length();
        }
        return Integer.MIN_VALUE;
    }

    private Object instantiateFieldObject(Field field,
                                          AssignmentNode node,
                                          ClassLoader loader) throws Exception {
        Class<?> fieldType = field.getType();

        if (node.value instanceof String && isSymbolicHandle((String) node.value)) {
            Object rebuilt = rebuildObjectFromHandle((String) node.value,
                    fieldType,
                    currentFieldValueContext == null ? Collections.emptyMap() : currentFieldValueContext,
                    loader);
            if (rebuilt != null) {
                return rebuilt;
            }
        }

        // First try to extract a type hint from strings such as "pkg.Class@xxxx".
        if (node.value instanceof String) {
            String hintedClassName = extractClassHint((String) node.value);
            if (hintedClassName != null) {
                try {
                    Class<?> hintedClass = Class.forName(hintedClassName, true, loader);
                    if (fieldType.isAssignableFrom(hintedClass)) {
                        return instantiateClass(hintedClass, flattenImmediateFieldMap(node), loader);
                    }
                } catch (Throwable ignored) {
                }
            }
        }

        if (Collection.class.isAssignableFrom(fieldType)) {
            return instantiateCollectionFieldObject(field, node, loader);
        }

        if (Map.class.isAssignableFrom(fieldType)) {
            return instantiateMapFieldObject(fieldType);
        }

        if (fieldType.isArray()) {
            return instantiateArrayFieldObject(fieldType.getComponentType(), node, loader);
        }

        Object special = instantiateSpecialAbstractType(fieldType, flattenLeafFieldMap(node), loader);
        if (special != null) {
            return special;
        }

        if (!fieldType.isInterface() && !Modifier.isAbstract(fieldType.getModifiers())) {
            return instantiateClass(fieldType, flattenImmediateFieldMap(node), loader);
        }

        return null;
    }

    private Object instantiateSpecialAbstractType(Class<?> targetType,
                                                  Map<String, Object> fieldValues,
                                                  ClassLoader loader) throws Exception {
        if (targetType == null) {
            return null;
        }

        if (OutputStream.class.isAssignableFrom(targetType)) {
            return new ByteArrayOutputStream();
        }
        if (InputStream.class.isAssignableFrom(targetType)) {
            return new ByteArrayInputStream(new byte[]{1});
        }

        String typeName = targetType.getName();
        if ("com.sun.net.httpserver.HttpExchange".equals(typeName)) {
            return createHttpExchangeDouble(fieldValues);
        }
        if ("com.haleywang.monitor.common.mvc.BaseCtrl".equals(typeName)) {
            return createOmegaBaseCtrlDouble(loader);
        }
        if ("me.chanjar.weixin.mp.config.WxMpConfigStorage".equals(typeName)) {
            return createWxMpConfigStorage(fieldValues, loader);
        }
        if ("me.chanjar.weixin.mp.api.WxMpService".equals(typeName)
                || "me.chanjar.weixin.common.util.http.RequestHttp".equals(typeName)) {
            return createWxMpServiceJoddHttp(fieldValues, loader);
        }
        if ("se.yolean.kafka.keyvalue.onupdate.DispatcherConfig".equals(typeName)
                || "se.yolean.kafka.keyvalue.onupdate.hc.RetryDecisions".equals(typeName)) {
            return createKafkaDispatcherConfig(fieldValues, loader);
        }

        if (!targetType.isInterface() && !Modifier.isAbstract(targetType.getModifiers())) {
            return null;
        }

        if ("com.lithium.flow.filer.Filer".equals(typeName)) {
            return createFlowFilerProxy(targetType, fieldValues, loader);
        }
        if ("com.lithium.flow.config.Config".equals(typeName)) {
            return createFlowConfigProxy(targetType, fieldValues, loader);
        }
        if ("java.util.function.Function".equals(typeName)) {
            return createFlowFunctionProxy(targetType, fieldValues, loader);
        }
        if ("com.lithium.flow.streams.Streamer".equals(typeName)) {
            return createFlowStreamerProxy(targetType, fieldValues, loader);
        }
        if ("com.lithium.flow.compress.Coder".equals(typeName)) {
            return createFlowCoderProxy(targetType, fieldValues, loader);
        }

        return null;
    }

    private Object createOmegaBaseCtrlDouble(ClassLoader loader) {
        try {
            Class<?> reqCtrl = Class.forName("com.haleywang.monitor.ctrl.v1.ReqCtrl", true, loader);
            return instantiateClass(reqCtrl, Collections.emptyMap(), loader);
        } catch (Throwable e) {
            log.info("[DEBUG createOmegaBaseCtrlDouble] failed to instantiate ReqCtrl: {}", e.toString());
            return null;
        }
    }

    private HttpExchange createHttpExchangeDouble(Map<String, Object> fieldValues) {
        SimpleHttpExchange exchange = new SimpleHttpExchange();
        Object body = firstPresent(fieldValues,
                "requestBody",
                "body",
                "getRequestBody",
                "requestBody.decoded(CHARSET)");
        if (body != null) {
            exchange.requestBody = String.valueOf(body);
        }

        Object uri = firstPresent(fieldValues, "requestURI", "requestUri", "uri");
        if (uri != null) {
            try {
                exchange.requestURI = URI.create(String.valueOf(uri));
            } catch (IllegalArgumentException ignored) {
            }
        }

        Object method = firstPresent(fieldValues, "requestMethod", "method");
        if (method != null) {
            exchange.requestMethod = String.valueOf(method);
        }
        return exchange;
    }

    private void applyOmegaTesterSemanticDefaults(Object target, ClassLoader loader) {
        if (target == null) {
            return;
        }
        String className = target.getClass().getName();
        if (!"com.haleywang.monitor.service.impl.ReqInfoServiceImpl".equals(className)) {
            return;
        }

        try {
            Field reqSettingService = tryFindField(target.getClass(), "reqSettingService");
            if (reqSettingService == null) {
                return;
            }
            reqSettingService.setAccessible(true);
            if (reqSettingService.get(target) != null) {
                return;
            }
            Object service = createOmegaReqSettingServiceProxy(loader);
            if (service != null) {
                reqSettingService.set(target, service);
                log.info("[DEBUG applyOmegaTesterSemanticDefaults] installed ReqSettingService proxy for {}",
                        className);
            }
        } catch (Throwable e) {
            log.info("[DEBUG applyOmegaTesterSemanticDefaults] failed for {}: {}", className, e.toString());
        }
    }

    private void applyNinjaCoreSemanticDefaults(Object target,
                                                Map<String, Object> fieldValues,
                                                ClassLoader loader) {
        if (target == null || !"ninja.AssetsController".equals(target.getClass().getName())) {
            return;
        }

        try {
            Field helperField = tryFindField(target.getClass(), "assetsControllerHelper");
            if (helperField != null) {
                helperField.setAccessible(true);
                if (helperField.get(target) == null) {
                    Class<?> helperClass = Class.forName("ninja.AssetsControllerHelper", true, loader);
                    helperField.set(target, instantiateClass(helperClass, Collections.emptyMap(), loader));
                }
            }

            Field propertiesField = tryFindField(target.getClass(), "ninjaProperties");
            if (propertiesField != null) {
                propertiesField.setAccessible(true);
                if (propertiesField.get(target) == null) {
                    Object proxy = createNinjaPropertiesProxy(fieldValues, loader);
                    if (proxy != null) {
                        propertiesField.set(target, proxy);
                    }
                }
            }
        } catch (Throwable e) {
            log.info("[DEBUG applyNinjaCoreSemanticDefaults] failed for {}: {}",
                    target.getClass().getName(), e.toString());
        }
    }

    private Object createNinjaPropertiesProxy(Map<String, Object> fieldValues, ClassLoader loader) {
        try {
            Class<?> propertiesType = Class.forName("ninja.utils.NinjaProperties", true, loader);
            boolean devMode = inferNinjaDevMode(fieldValues);
            return Proxy.newProxyInstance(loader, new Class[]{propertiesType}, (proxy, method, args) -> {
                if (isObjectMethod(method)) {
                    return handleObjectMethod(proxy, method, args, "NinjaPropertiesProxy");
                }

                String name = method.getName();
                if ("isDev".equals(name)) {
                    return devMode;
                }
                if ("isTest".equals(name)) {
                    return false;
                }
                if ("isProd".equals(name)) {
                    return !devMode;
                }
                if ("getContextPath".equals(name)) {
                    return "";
                }
                if ("setContextPath".equals(name)) {
                    return null;
                }
                if ("getAllCurrentNinjaProperties".equals(name)) {
                    return new Properties();
                }
                if (args != null && args.length >= 2 && name.endsWith("WithDefault")) {
                    return args[1];
                }
                if ("getStringArray".equals(name)) {
                    return new String[0];
                }
                return defaultValue(method.getReturnType());
            });
        } catch (Throwable e) {
            log.info("[DEBUG createNinjaPropertiesProxy] failed: {}", e.toString());
            return null;
        }
    }

    private boolean inferNinjaDevMode(Map<String, Object> fieldValues) {
        Object hint = firstPresent(fieldValues,
                "ninjaProperties.isDev()",
                "ninjaProperties.isDev",
                "ninjaMode",
                "this.ninjaProperties.isDev()",
                "this.ninjaProperties.isDev");
        if (hint == null) {
            return false;
        }
        if (hint instanceof Boolean) {
            return (Boolean) hint;
        }
        String text = String.valueOf(hint).trim();
        if ("dev".equalsIgnoreCase(text)) {
            return true;
        }
        return parseBoolean(text, false);
    }

    private void applyWxJavaSemanticDefaults(Object target,
                                             Map<String, Object> fieldValues,
                                             ClassLoader loader) {
        if (target == null || !target.getClass().getName().startsWith("me.chanjar.weixin.")) {
            return;
        }

        try {
            ensureWxMpServiceJoddDefaults(target, fieldValues, loader);
            installWxJavaRequestHttpIfPresent(target, fieldValues, loader);

            if ("me.chanjar.weixin.mp.api.impl.WxMpOAuth2ServiceImpl".equals(target.getClass().getName())) {
                Field serviceField = tryFindExactField(target.getClass(), "wxMpService");
                if (serviceField != null) {
                    serviceField.setAccessible(true);
                    Object current = serviceField.get(target);
                    if (!isInstanceOfClassName(current, "me.chanjar.weixin.mp.api.impl.BaseWxMpServiceImpl", loader)) {
                        current = createWxMpServiceJoddHttp(fieldValues, loader);
                        if (current != null) {
                            serviceField.set(target, current);
                        }
                    }
                    ensureWxMpServiceJoddDefaults(current, fieldValues, loader);
                }
            }
        } catch (Throwable e) {
            log.info("[DEBUG applyWxJavaSemanticDefaults] failed for {}: {}",
                    target.getClass().getName(), e.toString());
        }
    }

    private Object createWxMpServiceJoddHttp(Map<String, Object> fieldValues, ClassLoader loader) {
        try {
            Class<?> serviceClass = Class.forName(
                    "me.chanjar.weixin.mp.api.impl.WxMpServiceJoddHttpImpl", true, loader);
            Object service = instantiateClass(serviceClass,
                    fieldValues == null ? Collections.emptyMap() : fieldValues, loader);
            ensureWxMpServiceJoddDefaults(service, fieldValues, loader);
            return service;
        } catch (Throwable e) {
            log.info("[DEBUG createWxMpServiceJoddHttp] failed: {}", e.toString());
            return null;
        }
    }

    private void ensureWxMpServiceJoddDefaults(Object service,
                                               Map<String, Object> fieldValues,
                                               ClassLoader loader) {
        if (!isInstanceOfClassName(service, "me.chanjar.weixin.mp.api.impl.BaseWxMpServiceImpl", loader)) {
            return;
        }

        try {
            Field configMapField = tryFindExactField(service.getClass(), "configStorageMap");
            if (configMapField != null) {
                configMapField.setAccessible(true);
                Object configMap = configMapField.get(service);
                if (!(configMap instanceof Map) || ((Map<?, ?>) configMap).isEmpty()) {
                    Map<String, Object> map = new LinkedHashMap<>();
                    map.put("default", createWxMpConfigStorage(fieldValues, loader));
                    configMapField.set(service, map);
                }
            }

            Field httpClientField = tryFindExactField(service.getClass(), "httpClient");
            if (httpClientField != null) {
                httpClientField.setAccessible(true);
                if (httpClientField.get(service) == null) {
                    Class<?> providerClass = Class.forName(
                            "jodd.http.net.SocketHttpConnectionProvider", true, loader);
                    httpClientField.set(service, instantiateClass(providerClass, Collections.emptyMap(), loader));
                }
            }

            setWxMpConfigStorageHolderDefault(loader);
        } catch (Throwable e) {
            log.info("[DEBUG ensureWxMpServiceJoddDefaults] failed for {}: {}",
                    service == null ? "null" : service.getClass().getName(), e.toString());
        }
    }

    private Object createWxMpConfigStorage(Map<String, Object> fieldValues, ClassLoader loader) {
        try {
            Class<?> configClass = Class.forName(
                    "me.chanjar.weixin.mp.config.impl.WxMpDefaultConfigImpl", true, loader);
            Object config = instantiateClass(configClass, Collections.emptyMap(), loader);

            Map<String, Object> configFields = new LinkedHashMap<>();
            configFields.put("appId", inferWxJavaConfigValue(fieldValues, "appId", "wx-replay-app"));
            configFields.put("secret", inferWxJavaConfigValue(fieldValues, "secret", "s3cr3t"));
            configFields.put("accessToken", "expired-token");
            configFields.put("expiresTime", 0L);
            configFields.put("accessTokenLock", new java.util.concurrent.locks.ReentrantLock());
            applyFieldValues(config, configFields, loader);
            return config;
        } catch (Throwable e) {
            log.info("[DEBUG createWxMpConfigStorage] failed: {}", e.toString());
            return null;
        }
    }

    private Object inferWxJavaConfigValue(Map<String, Object> fieldValues,
                                          String token,
                                          String fallback) {
        Object explicit = findWxJavaConfigHint(fieldValues, token);
        if (explicit != null) {
            return explicit;
        }
        if ("appId".equals(token)) {
            Object payload = findWxJavaPayloadString(fieldValues);
            if (payload != null) {
                return payload;
            }
        }
        return fallback;
    }

    private Object findWxJavaConfigHint(Map<String, Object> fieldValues, String token) {
        if (fieldValues == null || fieldValues.isEmpty()) {
            return null;
        }
        String wanted = normalizeFieldLookupToken(token);
        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = normalizeFieldLookupToken(entry.getKey());
            if (key.contains(wanted)) {
                return entry.getValue();
            }
        }
        return null;
    }

    private Object findWxJavaPayloadString(Map<String, Object> fieldValues) {
        if (fieldValues == null || fieldValues.isEmpty()) {
            return null;
        }
        for (Object value : fieldValues.values()) {
            if (value == null) {
                continue;
            }
            String s = String.valueOf(value);
            if (looksLikeWxJavaInjectionPayload(s)) {
                return s;
            }
        }
        return null;
    }

    private boolean looksLikeWxJavaInjectionPayload(String value) {
        if (value == null) {
            return false;
        }
        String lower = value.toLowerCase(Locale.ROOT);
        return value.contains("\r")
                || value.contains("\n")
                || lower.contains("\\r")
                || lower.contains("\\n")
                || lower.contains("%0d")
                || lower.contains("%0a")
                || lower.contains("injected-header");
    }

    private void installWxJavaRequestHttpIfPresent(Object target,
                                                   Map<String, Object> fieldValues,
                                                   ClassLoader loader) {
        try {
            Field requestHttpField = tryFindExactField(target.getClass(), "requestHttp");
            if (requestHttpField == null) {
                return;
            }
            requestHttpField.setAccessible(true);
            if (requestHttpField.get(target) == null) {
                requestHttpField.set(target, createWxMpServiceJoddHttp(fieldValues, loader));
            }
        } catch (Throwable e) {
            log.info("[DEBUG installWxJavaRequestHttpIfPresent] failed for {}: {}",
                    target.getClass().getName(), e.toString());
        }
    }

    private boolean isInstanceOfClassName(Object value, String className, ClassLoader loader) {
        if (value == null) {
            return false;
        }
        try {
            Class<?> type = Class.forName(className, true, loader);
            return type.isInstance(value);
        } catch (Throwable ignored) {
            return false;
        }
    }

    private void setWxMpConfigStorageHolderDefault(ClassLoader loader) {
        try {
            Class<?> holder = Class.forName("me.chanjar.weixin.mp.util.WxMpConfigStorageHolder", true, loader);
            Method set = holder.getDeclaredMethod("set", String.class);
            set.setAccessible(true);
            set.invoke(null, "default");
        } catch (Throwable ignored) {
        }
    }

    private Object createOmegaReqSettingServiceProxy(ClassLoader loader) {
        try {
            Class<?> serviceType = Class.forName("com.haleywang.monitor.service.ReqSettingService", true, loader);
            return Proxy.newProxyInstance(loader, new Class[]{serviceType}, (proxy, method, args) -> {
                if (isObjectMethod(method)) {
                    return handleObjectMethod(proxy, method, args, "OmegaReqSettingServiceProxy");
                }
                if ("findByTypeAndOnwerAndCurrent".equals(method.getName())) {
                    return createOmegaReqSetting(loader);
                }
                if (List.class.isAssignableFrom(method.getReturnType())) {
                    return Collections.emptyList();
                }
                return defaultValue(method.getReturnType());
            });
        } catch (Throwable e) {
            log.info("[DEBUG createOmegaReqSettingServiceProxy] failed: {}", e.toString());
            return null;
        }
    }

    private void applyKafkaKeyvalueSemanticDefaults(Object target,
                                                    Map<String, Object> fieldValues,
                                                    ClassLoader loader) {
        if (target == null || !target.getClass().getName().startsWith("se.yolean.kafka.keyvalue.")) {
            return;
        }

        try {
            if ("se.yolean.kafka.keyvalue.onupdate.OnUpdateForwarder".equals(target.getClass().getName())) {
                installKafkaDispatcherConfig(target, fieldValues, loader);
                installKafkaTargetOptionals(target, fieldValues);
            } else if ("se.yolean.kafka.keyvalue.onupdate.hc.DispatcherConfigHttpclient"
                    .equals(target.getClass().getName())) {
                applyKafkaRetryDefaults(target, fieldValues);
            }
        } catch (Throwable e) {
            log.info("[DEBUG applyKafkaKeyvalueSemanticDefaults] failed for {}: {}",
                    target.getClass().getName(), e.toString());
        }
    }

    private Object createKafkaDispatcherConfig(Map<String, Object> fieldValues, ClassLoader loader) {
        try {
            Class<?> configClass = Class.forName(
                    "se.yolean.kafka.keyvalue.onupdate.hc.DispatcherConfigHttpclient", true, loader);
            Object config = instantiateClass(configClass, Collections.emptyMap(), loader);
            applyKafkaRetryDefaults(config, fieldValues);
            return config;
        } catch (Throwable e) {
            log.info("[DEBUG createKafkaDispatcherConfig] failed: {}", e.toString());
            return null;
        }
    }

    private void installKafkaDispatcherConfig(Object forwarder,
                                              Map<String, Object> fieldValues,
                                              ClassLoader loader) throws Exception {
        Field dispatcherConfig = tryFindExactField(forwarder.getClass(), "dispatcherConfig");
        if (dispatcherConfig == null) {
            return;
        }
        dispatcherConfig.setAccessible(true);
        Object current = dispatcherConfig.get(forwarder);
        if (current == null
                || !isInstanceOfClassName(current,
                "se.yolean.kafka.keyvalue.onupdate.hc.DispatcherConfigHttpclient", loader)) {
            dispatcherConfig.set(forwarder, createKafkaDispatcherConfig(fieldValues, loader));
        } else {
            applyKafkaRetryDefaults(current, fieldValues);
        }
    }

    private void installKafkaTargetOptionals(Object forwarder, Map<String, Object> fieldValues) throws Exception {
        String targetValue = inferKafkaKeyvalueTarget(fieldValues);
        for (int i = 0; i <= 9; i++) {
            String fieldName = i == 0 ? "target" : "target" + i;
            Field field = tryFindExactField(forwarder.getClass(), fieldName);
            if (field == null) {
                continue;
            }
            field.setAccessible(true);
            Object current = field.get(forwarder);
            if (current instanceof Optional) {
                continue;
            }
            if (i == 0 && targetValue != null) {
                field.set(forwarder, Optional.of(targetValue));
            } else {
                field.set(forwarder, Optional.empty());
            }
        }
    }

    private String inferKafkaKeyvalueTarget(Map<String, Object> fieldValues) {
        Object explicit = firstPresent(fieldValues,
                "target",
                "configuredTarget",
                "url",
                "toURI(configuredTarget)",
                "toURI(url)");
        String fromExplicit = firstKafkaUrlFromValue(explicit);
        if (fromExplicit != null) {
            return fromExplicit;
        }

        Object targetsConfig = firstPresent(fieldValues, "getTargetsConfig()", "getTargetsConfig");
        String fromTargetsConfig = firstKafkaUrlFromValue(targetsConfig);
        if (fromTargetsConfig != null) {
            return fromTargetsConfig;
        }

        if (fieldValues != null) {
            for (Object value : fieldValues.values()) {
                String url = firstKafkaUrlFromValue(value);
                if (url != null) {
                    return url;
                }
            }
        }
        return null;
    }

    private String firstKafkaUrlFromValue(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof Collection) {
            for (Object item : (Collection<?>) value) {
                String url = firstKafkaUrlFromValue(item);
                if (url != null) {
                    return url;
                }
            }
            return null;
        }
        String text = String.valueOf(value).trim();
        if (text.isEmpty()) {
            return null;
        }
        if (text.startsWith("[") && text.endsWith("]")) {
            try {
                List<?> parsed = objectMapper.readValue(text, List.class);
                return firstKafkaUrlFromValue(parsed);
            } catch (Exception ignored) {
            }
        }
        int start = firstHttpUrlOffset(text);
        if (start < 0) {
            return null;
        }
        int end = text.length();
        for (int i = start; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (Character.isWhitespace(ch) || ch == '"' || ch == '\'' || ch == ']' || ch == ',') {
                end = i;
                break;
            }
        }
        String url = text.substring(start, end);
        return url.contains("@") ? url : null;
    }

    private int firstHttpUrlOffset(String text) {
        if (text == null) {
            return -1;
        }
        int http = text.indexOf("http://");
        int https = text.indexOf("https://");
        if (http < 0) {
            return https;
        }
        if (https < 0) {
            return http;
        }
        return Math.min(http, https);
    }

    private void applyKafkaRetryDefaults(Object config, Map<String, Object> fieldValues) throws Exception {
        setIntFieldIfPresent(config,
                "maxRetriesConnectionRefused",
                firstPresent(fieldValues, "maxRetriesConnectionRefused", "dispatcherConfig.maxRetriesConnectionRefused"),
                8);
        setIntFieldIfPresent(config,
                "maxRetriesStatus",
                firstPresent(fieldValues, "maxRetriesStatus", "dispatcherConfig.maxRetriesStatus"),
                8);
    }

    private void setIntFieldIfPresent(Object target, String fieldName, Object rawValue, int fallback) throws Exception {
        if (target == null) {
            return;
        }
        Field field = tryFindExactField(target.getClass(), fieldName);
        if (field == null) {
            return;
        }
        field.setAccessible(true);
        int value = rawValue == null ? fallback : Integer.parseInt(String.valueOf(rawValue));
        field.setInt(target, value);
    }

    private Object createOmegaReqSetting(ClassLoader loader) {
        try {
            Class<?> settingType = Class.forName("com.haleywang.monitor.entity.ReqSetting", true, loader);
            Object setting = instantiateClass(settingType, Collections.emptyMap(), loader);
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put("content", "{}");
            applyFieldValues(setting, fields, loader);
            return setting;
        } catch (Throwable e) {
            log.info("[DEBUG createOmegaReqSetting] failed: {}", e.toString());
            return null;
        }
    }

    private Object firstPresent(Map<String, Object> values, String... keys) {
        if (values == null || keys == null) {
            return null;
        }
        for (String key : keys) {
            if (values.containsKey(key)) {
                return values.get(key);
            }
        }
        return null;
    }

    private Object createFlowConfigProxy(Class<?> targetType,
                                         Map<String, Object> fieldValues,
                                         ClassLoader loader) {
        Map<String, Object> hints = canonicalizeFlowHintMap(fieldValues);
        inferFlowConfigDefaults(hints);
        return Proxy.newProxyInstance(loader, new Class[]{targetType}, (proxy, method, args) -> {
            String name = method.getName();
            if (isObjectMethod(method)) {
                return handleObjectMethod(proxy, method, args, "FlowConfigProxy");
            }
            if (method.isDefault()) {
                return invokeDefaultInterfaceMethod(proxy, method, args);
            }

            String key = args != null && args.length > 0 && args[0] != null ? String.valueOf(args[0]) : "";
            switch (name) {
                case "getName":
                    return "replay";
                case "isAllowUndefined":
                    return true;
                case "containsKey":
                    return hints.containsKey(key);
                case "getRaw":
                case "getString":
                    return resolveConfigStringHint(hints, key, args);
                case "getInt":
                    return parseInt(resolveConfigStringHint(hints, key, args), defaultIntArg(args, 1));
                case "getLong":
                case "getTime":
                    return parseLong(resolveConfigStringHint(hints, key, args), defaultLongArg(args, 1));
                case "getDouble":
                    return parseDouble(resolveConfigStringHint(hints, key, args), defaultDoubleArg(args, 1));
                case "getBoolean":
                    return parseBoolean(resolveConfigStringHint(hints, key, args), defaultBooleanArg(args, 1));
                case "getList":
                    return resolveConfigListHint(hints, key, args);
                case "keySet":
                    return new LinkedHashSet<>(hints.keySet());
                case "asMap":
                case "asRawMap":
                    return new LinkedHashMap<>(stringifyMap(hints));
                default:
                    return defaultValue(method.getReturnType());
            }
        });
    }

    private Object createFlowFilerProxy(Class<?> targetType,
                                        Map<String, Object> fieldValues,
                                        ClassLoader loader) {
        Map<String, Object> hints = canonicalizeFlowHintMap(fieldValues);
        return Proxy.newProxyInstance(loader, new Class[]{targetType}, (proxy, method, args) -> {
            if (isObjectMethod(method)) {
                return handleObjectMethod(proxy, method, args, "FlowFilerProxy");
            }
            if (method.isDefault()) {
                return invokeDefaultInterfaceMethod(proxy, method, args);
            }

            String methodName = method.getName();
            switch (methodName) {
                case "getUri":
                    return URI.create("file:/");
                case "writeFile":
                case "appendFile":
                    return resolveFlowOutputStreamHint(hints, methodName, loader);
                case "readFile":
                    return resolveFlowInputStreamHint(hints, methodName, loader);
                case "listRecords":
                    return Collections.emptyList();
                case "getRecord":
                    return resolveFlowRecordHint(args, loader);
                case "createDirs":
                case "deleteFile":
                case "renameFile":
                case "setFileTime":
                case "deleteDir":
                case "close":
                    return null;
                case "openFile":
                    throw new UnsupportedOperationException("replay filer proxy does not implement openFile");
                default:
                    return defaultValue(method.getReturnType());
            }
        });
    }

    private Object createFlowFunctionProxy(Class<?> targetType,
                                           Map<String, Object> fieldValues,
                                           ClassLoader loader) {
        Map<String, Object> hints = canonicalizeFlowHintMap(fieldValues);
        return Proxy.newProxyInstance(loader, new Class[]{targetType}, (proxy, method, args) -> {
            if (isObjectMethod(method)) {
                return handleObjectMethod(proxy, method, args, "FlowFunctionProxy");
            }
            if (method.isDefault()) {
                return invokeDefaultInterfaceMethod(proxy, method, args);
            }
            if ("apply".equals(method.getName())) {
                Object exact = findFlowFunctionApplyHint(hints, args == null || args.length == 0 ? null : args[0]);
                if (exact != null || hasExplicitNullFunctionHint(hints, args == null || args.length == 0 ? null : args[0])) {
                    return normalizeFlowNullLike(exact);
                }
                return null;
            }
            return defaultValue(method.getReturnType());
        });
    }

    private Object createFlowStreamerProxy(Class<?> targetType,
                                           Map<String, Object> fieldValues,
                                           ClassLoader loader) {
        Map<String, Object> hints = canonicalizeFlowHintMap(fieldValues);
        return Proxy.newProxyInstance(loader, new Class[]{targetType}, (proxy, method, args) -> {
            if (isObjectMethod(method)) {
                return handleObjectMethod(proxy, method, args, "FlowStreamerProxy");
            }
            if (method.isDefault()) {
                return invokeDefaultInterfaceMethod(proxy, method, args);
            }
            if ("filterOut".equals(method.getName())) {
                if (flowHintRequestsNull(hints, "filterOut")) {
                    return null;
                }
                return args != null && args.length > 0 ? args[0] : new ByteArrayOutputStream();
            }
            if ("filterIn".equals(method.getName())) {
                return args != null && args.length > 0 ? args[0] : new ByteArrayInputStream(new byte[0]);
            }
            return defaultValue(method.getReturnType());
        });
    }

    private Object createFlowCoderProxy(Class<?> targetType,
                                        Map<String, Object> fieldValues,
                                        ClassLoader loader) {
        Map<String, Object> hints = canonicalizeFlowHintMap(fieldValues);
        return Proxy.newProxyInstance(loader, new Class[]{targetType}, (proxy, method, args) -> {
            if (isObjectMethod(method)) {
                return handleObjectMethod(proxy, method, args, "FlowCoderProxy");
            }
            if (method.isDefault()) {
                return invokeDefaultInterfaceMethod(proxy, method, args);
            }
            switch (method.getName()) {
                case "wrapOut":
                    return args != null && args.length > 0 ? args[0] : resolveFlowOutputStreamHint(hints, "wrapOut", loader);
                case "wrapIn":
                    return args != null && args.length > 0 ? args[0] : resolveFlowInputStreamHint(hints, "wrapIn", loader);
                case "getExtension":
                    return ".bin";
                default:
                    return defaultValue(method.getReturnType());
            }
        });
    }

    private Object instantiateCollectionFieldObject(Field field,
                                                    AssignmentNode node,
                                                    ClassLoader loader) throws Exception {
        Collection<Object> collection = createMutableCollection(field.getType());
        if (collection == null) {
            return null;
        }
        if (!(collection instanceof List<?>)) {
            populateCollectionFromHint(collection, resolveCollectionElementUpperBound(field), node.value, loader);
            return finalizeCollectionForFieldType(field.getType(), collection, loader);
        }

        @SuppressWarnings("unchecked")
        List<Object> list = (List<Object>) collection;
        Class<?> elementType = resolveCollectionElementUpperBound(field);
        int maxIndex = maxIndexedChild(node);
        if (maxIndex < 0) {
            populateCollectionFromHint(list, elementType, node.value, loader);
            return finalizeCollectionForFieldType(field.getType(), list, loader);
        }

        List<String> hintedHandles = extractHandleSequence(node.value);
        for (int i = 0; i <= maxIndex; i++) {
            AssignmentNode child = node.children.get("[" + i + "]");
            if (i < hintedHandles.size() && (child == null || child.value == null)) {
                AssignmentNode synthetic = new AssignmentNode();
                synthetic.value = hintedHandles.get(i);
                if (child != null) {
                    synthetic.children.putAll(child.children);
                }
                child = synthetic;
            }
            if (child == null) {
                list.add(null);
                continue;
            }
            list.add(createIndexedContainerElement(elementType, child, loader));
        }
        return finalizeCollectionForFieldType(field.getType(), list, loader);
    }

    private Object instantiateMapFieldObject(Class<?> fieldType) {
        return defaultMapForType(fieldType);
    }

    private Object instantiateArrayFieldObject(Class<?> componentType,
                                               AssignmentNode node,
                                               ClassLoader loader) throws Exception {
        int maxIndex = maxIndexedChild(node);
        if (maxIndex < 0) {
            return Array.newInstance(componentType, 0);
        }

        Object array = Array.newInstance(componentType, maxIndex + 1);
        for (int i = 0; i <= maxIndex; i++) {
            AssignmentNode child = node.children.get("[" + i + "]");
            if (child == null) {
                continue;
            }
            Object element = createIndexedContainerElement(componentType, child, loader);
            if (element != null) {
                Array.set(array, i, element);
            }
        }
        return array;
    }

    private int maxIndexedChild(AssignmentNode node) {
        int max = -1;
        if (node == null) {
            return max;
        }

        for (String key : node.children.keySet()) {
            if (!isBracketToken(key)) {
                continue;
            }
            String inside = key.substring(1, key.length() - 1).trim();
            if (!inside.matches("\\d+")) {
                continue;
            }
            max = Math.max(max, Integer.parseInt(inside));
        }
        return max;
    }

    private Object createIndexedContainerElement(Class<?> elementType,
                                                 AssignmentNode child,
                                                 ClassLoader loader) throws Exception {
        if (child == null) {
            return null;
        }

        if (child.value instanceof String && isSymbolicHandle((String) child.value)) {
            Object rebuilt = rebuildObjectFromHandle((String) child.value,
                    elementType == null ? Object.class : elementType,
                    currentFieldValueContext == null ? Collections.emptyMap() : currentFieldValueContext,
                    loader);
            if (rebuilt != null) {
                if (!child.children.isEmpty()) {
                    applyAssignmentNode(rebuilt, child, loader, "(collectionElement)");
                }
                return rebuilt;
            }
        }

        if (child.children.isEmpty()) {
            if (elementType == null || elementType == Object.class) {
                return tryCreateDynamicValue(child, loader);
            }
            return convertValueOrTreatAsHint(child.value, elementType, loader);
        }

        if (elementType != null
                && elementType != Object.class
                && !elementType.isInterface()
                && !Modifier.isAbstract(elementType.getModifiers())) {
            Object element = instantiateClass(elementType, flattenImmediateFieldMap(child), loader);
            if (element != null && !child.children.isEmpty()) {
                applyAssignmentNode(element, child, loader, "(collectionElement)");
            }
            return element;
        }

        Class<?> inferredElementType = inferCollectionElementConcreteType(elementType, child, loader);
        if (inferredElementType != null) {
            if (child.value instanceof String && isSymbolicHandle((String) child.value)) {
                Object rebuilt = rebuildObjectFromHandle((String) child.value,
                        inferredElementType,
                        currentFieldValueContext == null ? Collections.emptyMap() : currentFieldValueContext,
                        loader);
                if (rebuilt != null) {
                    if (!child.children.isEmpty()) {
                        applyAssignmentNode(rebuilt, child, loader, "(collectionElement)");
                    }
                    return rebuilt;
                }
            }
            Object element = instantiateClass(inferredElementType, flattenImmediateFieldMap(child), loader);
            if (element != null && !child.children.isEmpty()) {
                applyAssignmentNode(element, child, loader, "(collectionElement)");
            }
            return element;
        }

        return tryCreateDynamicValue(child, loader);
    }

    private Class<?> inferCollectionElementConcreteType(Class<?> elementType,
                                                        AssignmentNode child,
                                                        ClassLoader loader) {
        if (elementType == null || elementType == Object.class || child == null) {
            return null;
        }

        if (child.value instanceof String) {
            String hintedClassName = extractClassHint((String) child.value);
            if (hintedClassName != null) {
                try {
                    Class<?> hintedClass = Class.forName(hintedClassName, true, loader);
                    if (elementType.isAssignableFrom(hintedClass)
                            && !hintedClass.isInterface()
                            && !Modifier.isAbstract(hintedClass.getModifiers())) {
                        return hintedClass;
                    }
                } catch (Throwable ignored) {
                }
            }
        }

        if (!elementType.isInterface() && !Modifier.isAbstract(elementType.getModifiers())) {
            return elementType;
        }

        Set<Class<?>> candidates = discoverConcreteCandidates(elementType, elementType, loader);
        int bestScore = Integer.MIN_VALUE;
        Class<?> best = null;
        for (Class<?> candidate : candidates) {
            int score = scoreCandidate(candidate, child.children);
            if (score > bestScore) {
                bestScore = score;
                best = candidate;
            }
        }
        return bestScore > 0 ? best : null;
    }

    private String extractClassHint(String raw) {
        if (raw == null) {
            return null;
        }
        String s = raw.trim();
        int at = s.indexOf('@');
        if (at <= 0) {
            return null;
        }
        String className = s.substring(0, at);
        if (!className.contains(".")) {
            return null;
        }
        return className;
    }

    private boolean isBracketToken(String token) {
        return token != null && token.startsWith("[") && token.endsWith("]");
    }

    private boolean applyBracketAssignment(Object obj,
                                           String childName,
                                           AssignmentNode childNode,
                                           ClassLoader loader,
                                           String objectPath) throws Exception {
        String inside = childName.substring(1, childName.length() - 1).trim();

        if (obj instanceof Map<?, ?>) {
            return applyMapKeyAssignment(castMap(obj), canonicalizeWellKnownLiteral(inside), childNode, loader, objectPath);
        }

        if (!(inside.matches("\\d+"))) {
            return false;
        }

        int index = Integer.parseInt(inside);

        if (obj instanceof List<?>) {
            @SuppressWarnings("unchecked")
            List<Object> list = (List<Object>) obj;
            try {
                ensureListSize(list, index + 1);
            } catch (UnsupportedOperationException ignored) {
                if (index >= list.size()) {
                    return false;
                }
            }
            if (childNode.children.isEmpty()) {
                Object replacement = tryCreateDynamicValue(childNode, loader);
                try {
                    list.set(index, replacement);
                } catch (UnsupportedOperationException e) {
                    if (index >= list.size()) {
                        return false;
                    }
                    Object current = list.get(index);
                    if (current != null) {
                        if (childNode.value instanceof String && isSymbolicHandle((String) childNode.value)) {
                            Map<String, Object> nestedHints = expandHandleScopedFieldValues((String) childNode.value,
                                    currentFieldValueContext == null ? Collections.emptyMap() : currentFieldValueContext,
                                    new LinkedHashSet<>());
                            if (!nestedHints.isEmpty()) {
                                applyHandleAwareFieldValues(current, nestedHints, loader);
                            }
                        }
                        return true;
                    }
                    return false;
                }
                return true;
            }

            Object current = index < list.size() ? list.get(index) : null;
            if (current == null) {
                current = tryCreateDynamicValue(childNode, loader);
                try {
                    list.set(index, current);
                } catch (UnsupportedOperationException e) {
                    if (index >= list.size()) {
                        return false;
                    }
                    current = list.get(index);
                    if (current == null) {
                        return false;
                    }
                }
            }
            applyAssignmentNode(current, childNode, loader, objectPath + childName);
            return true;
        }

        if (obj.getClass().isArray()) {
            if (index >= Array.getLength(obj)) {
                return false;
            }

            Class<?> componentType = obj.getClass().getComponentType();
            if (childNode.children.isEmpty()) {
                Array.set(obj, index, convertValueOrTreatAsHint(childNode.value, componentType, loader));
                return true;
            }

            Object current = Array.get(obj, index);
            if (current == null
                    && !componentType.isInterface()
                    && !Modifier.isAbstract(componentType.getModifiers())) {
                current = instantiateClass(componentType, flattenImmediateFieldMap(childNode), loader);
                Array.set(obj, index, current);
            }
            if (current == null) {
                return false;
            }
            applyAssignmentNode(current, childNode, loader, objectPath + childName);
            return true;
        }

        if (obj instanceof Collection<?>) {
            Collection<Object> collection = castCollection(obj);
            List<Object> list = new ArrayList<>(collection);
            ensureListSize(list, index + 1);
            if (childNode.children.isEmpty()) {
                list.set(index, tryCreateDynamicValue(childNode, loader));
            } else {
                Object current = list.get(index);
                if (current == null) {
                    current = tryCreateDynamicValue(childNode, loader);
                    list.set(index, current);
                }
                applyAssignmentNode(current, childNode, loader, objectPath + childName);
            }
            collection.clear();
            collection.addAll(list);
            return true;
        }

        return false;
    }

    private boolean applyMapChildAssignment(Object obj,
                                            String childName,
                                            AssignmentNode childNode,
                                            ClassLoader loader,
                                            String objectPath) throws Exception {
        return applyMapChildAssignment(obj, childName, childNode, loader, objectPath, null);
    }

    private boolean applyMapChildAssignment(Object obj,
                                            String childName,
                                            AssignmentNode childNode,
                                            ClassLoader loader,
                                            String objectPath,
                                            Class<?> expectedValueTypeHint) throws Exception {
        return applyMapKeyAssignment(castMap(obj),
                canonicalizeWellKnownLiteral(childName),
                childNode,
                loader,
                objectPath,
                expectedValueTypeHint);
    }

    private boolean applyMapKeyAssignment(Map<Object, Object> map,
                                          Object key,
                                          AssignmentNode childNode,
                                          ClassLoader loader,
                                          String objectPath) throws Exception {
        return applyMapKeyAssignment(map, key, childNode, loader, objectPath, null);
    }

    private boolean applyMapKeyAssignment(Map<Object, Object> map,
                                          Object key,
                                          AssignmentNode childNode,
                                          ClassLoader loader,
                                          String objectPath,
                                          Class<?> expectedValueTypeHint) throws Exception {
        Class<?> expectedType = expectedValueTypeHint != null
                ? expectedValueTypeHint
                : resolveMapValueTargetType(key, loader);
        if (expectedType == null) {
            expectedType = inferExistingMapValueType(map);
        }
        if (childNode.children.isEmpty()) {
            Object value = expectedType == null
                    ? tryCreateDynamicValue(childNode, loader)
                    : convertValueOrTreatAsHint(childNode.value, expectedType, loader);
            map.put(key, value);
            log.info("[DEBUG applyFieldValues] set map {}[{}] = {}", objectPath, key, map.get(key));
            return true;
        }

        Object current = map.get(key);
        if (current == null) {
            current = createExpectedMapValue(childNode, expectedType, loader);
            if (current != null) {
                map.put(key, current);
            }
        }
        if (current == null) {
            current = tryCreateDynamicValue(childNode, loader);
            map.put(key, current);
        }
        if (current == null) {
            return false;
        }

        applyAssignmentNode(current, childNode, loader, objectPath + "[" + key + "]");
        return true;
    }

    private void applyTypedMapAssignmentNode(Map<Object, Object> map,
                                             AssignmentNode node,
                                             Class<?> expectedValueType,
                                             ClassLoader loader,
                                             String objectPath) throws Exception {
        if (map == null || node == null) {
            return;
        }

        Map<String, AssignmentNode> unmatched = new LinkedHashMap<>();
        for (Map.Entry<String, AssignmentNode> entry : node.children.entrySet()) {
            String childName = entry.getKey();
            AssignmentNode childNode = entry.getValue();

            if (isBracketToken(childName)) {
                if (applyMapKeyAssignment(map,
                        canonicalizeWellKnownLiteral(childName.substring(1, childName.length() - 1).trim()),
                        childNode,
                        loader,
                        objectPath,
                        expectedValueType)) {
                    continue;
                }
                unmatched.put(childName, childNode);
                continue;
            }

            CollapsedMapKey collapsed = collapseMapKeyChain(childName, childNode);
            String effectiveName = collapsed == null ? childName : collapsed.key;
            AssignmentNode effectiveNode = collapsed == null ? childNode : collapsed.node;
            if (applyMapChildAssignment(map, effectiveName, effectiveNode, loader, objectPath, expectedValueType)) {
                continue;
            }
            unmatched.put(effectiveName, effectiveNode);
        }

        if (!unmatched.isEmpty()) {
            boolean routed = routeUnmatchedToCollectionChildren(map, unmatched, loader, objectPath);
            if (!routed) {
                for (String name : unmatched.keySet()) {
                    log.info("[DEBUG applyFieldValues] unmatched path {}.{} ignored",
                            objectPath, name);
                }
            }
        }
    }

    private Class<?> resolveMapValueTargetType(Object key, ClassLoader loader) {
        if (!(key instanceof String)) {
            return null;
        }
        return resolveKnownMapValueType(canonicalizeWellKnownLiteral((String) key), loader);
    }

    private Object createExpectedMapValue(AssignmentNode childNode,
                                          Class<?> expectedType,
                                          ClassLoader loader) throws Exception {
        if (expectedType == null) {
            return null;
        }

        if (childNode.value != null) {
            try {
                return convertValueOrTreatAsHint(childNode.value, expectedType, loader);
            } catch (Exception ignored) {
            }
        }

        if (expectedType == Object.class) {
            return tryCreateDynamicValue(childNode, loader);
        }
        if (Map.class.isAssignableFrom(expectedType)) {
            return defaultMapForType(expectedType);
        }
        if (Collection.class.isAssignableFrom(expectedType)) {
            return defaultCollectionForType(expectedType);
        }
        if (expectedType.isArray()) {
            return instantiateArrayFieldObject(expectedType.getComponentType(), childNode, loader);
        }
        if (!expectedType.isInterface() && !Modifier.isAbstract(expectedType.getModifiers())) {
            return instantiateClass(expectedType, flattenImmediateFieldMap(childNode), loader);
        }

        return null;
    }

    private CollapsedMapKey collapseMapKeyChain(String firstToken, AssignmentNode firstNode) {
        if (firstToken == null || firstNode == null || isBracketToken(firstToken)) {
            return null;
        }

        List<String> parts = new ArrayList<>();
        parts.add(firstToken);

        AssignmentNode current = firstNode;
        String bestKey = null;
        AssignmentNode bestNode = null;

        while (true) {
            String candidate = String.join(".", parts);
            if (isLikelyQualifiedMapKey(candidate)) {
                bestKey = canonicalizeWellKnownLiteral(candidate);
                bestNode = current;
            }

            if (current.children.size() != 1) {
                break;
            }

            Map.Entry<String, AssignmentNode> child = current.children.entrySet().iterator().next();
            if (isBracketToken(child.getKey())) {
                break;
            }

            parts.add(child.getKey());
            current = child.getValue();
        }

        if (bestKey == null || parts.size() <= 1) {
            return null;
        }

        CollapsedMapKey collapsed = new CollapsedMapKey();
        collapsed.key = bestKey;
        collapsed.node = bestNode;
        return collapsed;
    }

    private boolean isLikelyQualifiedMapKey(String candidate) {
        if (candidate == null || candidate.isEmpty()) {
            return false;
        }

        String canonical = canonicalizeWellKnownLiteral(candidate);
        if (!Objects.equals(canonical, candidate)) {
            return true;
        }

        return candidate.startsWith("java.")
                || candidate.startsWith("javax.")
                || candidate.startsWith("jakarta.")
                || candidate.startsWith("org.")
                || candidate.startsWith("com.")
                || candidate.startsWith("net.")
                || candidate.startsWith("io.")
                || candidate.startsWith("sun.")
                || candidate.startsWith("Validator.");
    }

    @SuppressWarnings("unchecked")
    private Map<Object, Object> castMap(Object obj) {
        return (Map<Object, Object>) obj;
    }

    @SuppressWarnings("unchecked")
    private Collection<Object> castCollection(Object obj) {
        return (Collection<Object>) obj;
    }

    private void ensureListSize(List<Object> list, int size) {
        while (list.size() < size) {
            list.add(null);
        }
    }

    private Object tryCreateDynamicValue(AssignmentNode node, ClassLoader loader) throws Exception {
        if (node == null) {
            return null;
        }

        if (node.value instanceof String) {
            String raw = ((String) node.value).trim();
            String hintedClassName = extractClassHint(raw);
            if (hintedClassName != null) {
                try {
                    Class<?> hintedClass = Class.forName(hintedClassName, true, loader);
                    if (hintedClass == String.class) {
                        return "";
                    }
                    if (hintedClass == Class.class) {
                        return String.class;
                    }
                    if (Map.class.isAssignableFrom(hintedClass)) {
                        return defaultMapForType(hintedClass);
                    }
                    if (Collection.class.isAssignableFrom(hintedClass)) {
                        return defaultCollectionForType(hintedClass);
                    }
                    if (!hintedClass.isInterface() && !Modifier.isAbstract(hintedClass.getModifiers())) {
                        return instantiateClass(hintedClass, flattenImmediateFieldMap(node), loader);
                    }
                } catch (Throwable ignored) {
                }
            }

            if (raw.startsWith("{")) {
                try {
                    return new LinkedHashMap<>(objectMapper.readValue(raw, Map.class));
                } catch (Exception ignored) {
                }
            }
            if (raw.startsWith("[")) {
                try {
                    return new ArrayList<>(objectMapper.readValue(raw, List.class));
                } catch (Exception ignored) {
                }
            }
        }

        if (!node.children.isEmpty()) {
            boolean hasBracketChild = false;
            boolean allBracketChildrenNumeric = true;
            for (String key : node.children.keySet()) {
                if (isBracketToken(key)) {
                    hasBracketChild = true;
                    String inside = key.substring(1, key.length() - 1).trim();
                    if (!inside.matches("\\d+")) {
                        allBracketChildrenNumeric = false;
                    }
                    continue;
                }
                allBracketChildrenNumeric = false;
            }

            if (hasBracketChild && allBracketChildrenNumeric) {
                List<Object> list = new ArrayList<>();
                int max = maxIndexedChild(node);
                ensureListSize(list, max + 1);
                return list;
            }

            if (hasBracketChild) {
                return new LinkedHashMap<>();
            }

            return new LinkedHashMap<>();
        }

        return node.value;
    }

    private String canonicalizeWellKnownLiteral(String raw) {
        if (raw == null) {
            return null;
        }
        String value = raw.trim();
        switch (value) {
            case "Validator.BEAN_PARAM":
            case "org.apache.commons.validator.Validator.BEAN_PARAM":
            case "Validator.java.lang.Object":
            case "BEAN_PARAM":
            case "bean":
            case "java.lang.String[bean]":
                return "java.lang.Object";
            case "Validator.FIELD_PARAM":
            case "org.apache.commons.validator.Validator.FIELD_PARAM":
            case "Validator.org.apache.commons.validator.Field":
            case "FIELD_PARAM":
            case "field":
            case "java.lang.String[field]":
                return "org.apache.commons.validator.Field";
            case "Validator.VALIDATOR_ACTION_PARAM":
            case "org.apache.commons.validator.Validator.VALIDATOR_ACTION_PARAM":
            case "Validator.org.apache.commons.validator.ValidatorAction":
            case "VALIDATOR_ACTION_PARAM":
                return "org.apache.commons.validator.ValidatorAction";
            case "Validator.VALIDATOR_RESULTS_PARAM":
            case "org.apache.commons.validator.Validator.VALIDATOR_RESULTS_PARAM":
            case "Validator.org.apache.commons.validator.ValidatorResults":
            case "VALIDATOR_RESULTS_PARAM":
                return "org.apache.commons.validator.ValidatorResults";
            case "Validator.FORM_PARAM":
            case "org.apache.commons.validator.Validator.FORM_PARAM":
            case "Validator.org.apache.commons.validator.Form":
            case "FORM_PARAM":
                return "org.apache.commons.validator.Form";
            case "Validator.VALIDATOR_PARAM":
            case "org.apache.commons.validator.Validator.VALIDATOR_PARAM":
            case "Validator.org.apache.commons.validator.Validator":
            case "VALIDATOR_PARAM":
                return "org.apache.commons.validator.Validator";
            case "Validator.LOCALE_PARAM":
            case "org.apache.commons.validator.Validator.LOCALE_PARAM":
            case "Validator.java.util.Locale":
            case "LOCALE_PARAM":
                return "java.util.Locale";
            default:
                return value;
        }
    }

    private void applyStaticFieldLikeValues(Map<String, Object> fieldValues,
                                            Class<?> entryClass,
                                            ClassLoader loader) throws Exception {
        if (fieldValues == null || fieldValues.isEmpty()) {
            return;
        }

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String rawKey = entry.getKey();
            if (!looksLikeStaticClassPath(rawKey) || rawKey.contains("(")) {
                continue;
            }

            StaticFieldTarget target = resolveStaticFieldTarget(rawKey, entryClass, loader);
            if (target == null) {
                continue;
            }

            Field staticField = tryFindField(target.ownerClass, target.fieldName);
            if (staticField == null || !Modifier.isStatic(staticField.getModifiers())) {
                continue;
            }

            staticField.setAccessible(true);
            Object convertedValue = convertValueOrTreatAsHint(entry.getValue(), staticField.getType(), loader);
            staticField.set(null, convertedValue);
            log.info("[DEBUG applyStaticFieldLikeValues] set {}.{} = {}",
                    target.ownerClass.getName(), target.fieldName, convertedValue);
        }
    }

    private StaticFieldTarget resolveStaticFieldTarget(String rawKey,
                                                       Class<?> entryClass,
                                                       ClassLoader loader) {
        String key = normalizePath(rawKey);
        List<String> tokens = splitPathTokens(key);
        if (tokens.size() < 2) {
            return null;
        }

        for (int prefixSize = tokens.size() - 1; prefixSize >= 1; prefixSize--) {
            String classPart = String.join(".", tokens.subList(0, prefixSize));
            Class<?> ownerClass = tryResolveStaticOwnerClass(classPart, entryClass, loader);
            if (ownerClass == null) {
                continue;
            }

            if (tokens.size() - prefixSize != 1) {
                continue;
            }

            StaticFieldTarget target = new StaticFieldTarget();
            target.ownerClass = ownerClass;
            target.fieldName = tokens.get(prefixSize);
            return target;
        }

        return null;
    }

    private Class<?> tryResolveStaticOwnerClass(String classPart,
                                                Class<?> entryClass,
                                                ClassLoader loader) {
        if (classPart == null || classPart.isEmpty()) {
            return null;
        }

        try {
            return Class.forName(classPart, true, loader);
        } catch (Throwable ignored) {
        }

        if (entryClass != null && entryClass.getPackage() != null) {
            String fqcn = entryClass.getPackage().getName() + "." + classPart;
            try {
                return Class.forName(fqcn, true, loader);
            } catch (Throwable ignored) {
            }
        }

        return null;
    }

    private Map<String, Object> canonicalizeFlowHintMap(Map<String, Object> fieldValues) {
        Map<String, Object> canonical = new LinkedHashMap<>();
        if (fieldValues == null) {
            return canonical;
        }
        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            if (entry.getKey() == null) {
                continue;
            }
            canonical.put(entry.getKey(), entry.getValue());
            canonical.put(normalizeFlowHintKey(entry.getKey()), entry.getValue());
        }
        return canonical;
    }

    private Map<String, Object> enrichFlowFieldAliases(Map<String, Object> fieldValues,
                                                       ClassLoader loader) {
        Map<String, Object> enriched = new LinkedHashMap<>();
        if (fieldValues == null) {
            return enriched;
        }
        enriched.putAll(fieldValues);
        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (key == null) {
                continue;
            }
            if (key.startsWith("this.")) {
                enriched.putIfAbsent(key.substring("this.".length()), entry.getValue());
            }

            Matcher castedThis = Pattern.compile("^\\(\\([^)]*\\)(this(?:\\.[^)]+)?)\\)\\.(.+)$").matcher(key);
            if (castedThis.matches()) {
                enriched.putIfAbsent(castedThis.group(1) + "." + castedThis.group(2), entry.getValue());
                if (castedThis.group(1).startsWith("this.")) {
                    enriched.putIfAbsent(castedThis.group(1).substring("this.".length()) + "." + castedThis.group(2), entry.getValue());
                }
            }

            Matcher castedGeneric = Pattern.compile("^\\(\\([^)]*\\)([^)]+)\\)\\.(.+)$").matcher(key);
            if (castedGeneric.matches()) {
                enriched.putIfAbsent(castedGeneric.group(1) + "." + castedGeneric.group(2), entry.getValue());
            }

            Matcher simpleHandle = Pattern.compile("^([A-Za-z_$][A-Za-z0-9_$]*@[A-Za-z0-9]+)\\.(.+)$").matcher(key);
            if (simpleHandle.matches()) {
                enriched.putIfAbsent(simpleHandle.group(2), entry.getValue());
            }

            String strippedClassPrefix = stripFlowClassPrefixAlias(key, loader);
            if (strippedClassPrefix != null && !strippedClassPrefix.isEmpty()) {
                enriched.putIfAbsent(strippedClassPrefix, entry.getValue());
            }
        }
        return enriched;
    }

    private String stripFlowClassPrefixAlias(String rawKey, ClassLoader loader) {
        if (rawKey == null || rawKey.startsWith("this.") || rawKey.startsWith("arg")) {
            return null;
        }

        List<String> tokens = splitPathTokens(rawKey);
        if (tokens.size() < 2) {
            return null;
        }

        for (int prefixSize = tokens.size() - 1; prefixSize >= 1; prefixSize--) {
            String prefix = String.join(".", tokens.subList(0, prefixSize));
            if (looksResolvableFlowClassToken(prefix, loader)) {
                return String.join(".", tokens.subList(prefixSize, tokens.size()));
            }
        }

        String first = tokens.get(0);
        if (looksLikeFlowSimpleClassAlias(first)) {
            return String.join(".", tokens.subList(1, tokens.size()));
        }
        return null;
    }

    private boolean looksResolvableFlowClassToken(String token, ClassLoader loader) {
        if (token == null || token.isEmpty() || token.indexOf('[') >= 0) {
            return false;
        }
        try {
            Class<?> cls = Class.forName(token, false, loader);
            return cls.getName().startsWith("com.lithium.flow.");
        } catch (Throwable ignored) {
            return false;
        }
    }

    private boolean looksLikeFlowSimpleClassAlias(String token) {
        if (token == null || token.isEmpty()) {
            return false;
        }
        String simple = token;
        int at = simple.indexOf('@');
        if (at > 0) {
            simple = simple.substring(0, at);
        }
        return Character.isUpperCase(simple.charAt(0));
    }

    private String normalizeFlowHintKey(String raw) {
        if (raw == null) {
            return "";
        }
        return raw.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "");
    }

    private void inferFlowConfigDefaults(Map<String, Object> hints) {
        if (hints == null) {
            return;
        }

        hints.putIfAbsent("filer.url", "file:/tmp/vulnseer-flow-replay");

        boolean mentionsCompress = hints.keySet().stream()
                .map(this::normalizeFlowHintKey)
                .anyMatch(k -> k.contains("compressoption")
                        || k.contains("outcoder")
                        || k.contains("snappycoder"));
        if (mentionsCompress && !hints.containsKey("filer.streamers")) {
            hints.put("filer.streamers", Collections.singletonList("compress"));
        }
        if (mentionsCompress && !hints.containsKey("filer.compress.type")) {
            hints.put("filer.compress.type", "snappy");
        }
        if (!hints.containsKey("filer.compress.option")) {
            for (Map.Entry<String, Object> entry : hints.entrySet()) {
                if (normalizeFlowHintKey(entry.getKey()).contains("compressoption")) {
                    hints.put("filer.compress.option", entry.getValue());
                    break;
                }
            }
        }

        Object csvPath = firstNonNull(hints.get("csv.path"), hints.get("config.csv.path"));
        if (csvPath instanceof String) {
            ensureParentDirectory((String) csvPath);
        }
    }

    private String resolveConfigStringHint(Map<String, Object> hints, String key, Object[] args) {
        Object exact = firstNonNull(hints.get(key), hints.get("config." + key), hints.get(normalizeFlowHintKey(key)));
        if (exact == null && args != null && args.length > 1) {
            Object def = args[1];
            return def == null ? null : String.valueOf(def);
        }
        Object normalized = normalizeFlowNullLike(exact);
        return normalized == null ? null : String.valueOf(normalized);
    }

    private List<String> resolveConfigListHint(Map<String, Object> hints, String key, Object[] args) {
        Object exact = firstNonNull(hints.get(key), hints.get("config." + key), hints.get(normalizeFlowHintKey(key)));
        if (exact == null && args != null && args.length > 1 && args[1] instanceof List<?>) {
            @SuppressWarnings("unchecked")
            List<String> def = (List<String>) args[1];
            return def;
        }
        if (exact instanceof List<?>) {
            List<String> values = new ArrayList<>();
            for (Object value : (List<?>) exact) {
                values.add(String.valueOf(value));
            }
            return values;
        }
        if (exact instanceof String) {
            String s = ((String) exact).trim();
            if (s.isEmpty()) {
                return Collections.emptyList();
            }
            return Arrays.asList(s.split("\\s*,\\s*"));
        }
        return Collections.emptyList();
    }

    private Integer defaultIntArg(Object[] args, int index) {
        if (args != null && args.length > index && args[index] instanceof Number) {
            return ((Number) args[index]).intValue();
        }
        return 0;
    }

    private Long defaultLongArg(Object[] args, int index) {
        if (args != null && args.length > index && args[index] instanceof Number) {
            return ((Number) args[index]).longValue();
        }
        return 0L;
    }

    private Double defaultDoubleArg(Object[] args, int index) {
        if (args != null && args.length > index && args[index] instanceof Number) {
            return ((Number) args[index]).doubleValue();
        }
        return 0.0d;
    }

    private Boolean defaultBooleanArg(Object[] args, int index) {
        if (args != null && args.length > index && args[index] instanceof Boolean) {
            return (Boolean) args[index];
        }
        return Boolean.FALSE;
    }

    private int parseInt(String raw, Integer def) {
        if (raw == null) {
            return def == null ? 0 : def;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (Exception ignored) {
            return def == null ? 0 : def;
        }
    }

    private long parseLong(String raw, Long def) {
        if (raw == null) {
            return def == null ? 0L : def;
        }
        try {
            String digits = raw.trim().replaceAll("[^0-9-]", "");
            return digits.isEmpty() ? (def == null ? 0L : def) : Long.parseLong(digits);
        } catch (Exception ignored) {
            return def == null ? 0L : def;
        }
    }

    private double parseDouble(String raw, Double def) {
        if (raw == null) {
            return def == null ? 0.0d : def;
        }
        try {
            return Double.parseDouble(raw.trim());
        } catch (Exception ignored) {
            return def == null ? 0.0d : def;
        }
    }

    private boolean parseBoolean(String raw, Boolean def) {
        if (raw == null) {
            return def != null && def;
        }
        return Boolean.parseBoolean(raw.trim());
    }

    private Map<String, String> stringifyMap(Map<String, Object> hints) {
        Map<String, String> result = new LinkedHashMap<>();
        if (hints == null) {
            return result;
        }
        for (Map.Entry<String, Object> entry : hints.entrySet()) {
            if (entry.getValue() != null) {
                result.put(entry.getKey(), String.valueOf(entry.getValue()));
            }
        }
        return result;
    }

    private boolean isObjectMethod(Method method) {
        return method != null
                && method.getDeclaringClass() == Object.class
                && Arrays.asList("toString", "hashCode", "equals").contains(method.getName());
    }

    private Object handleObjectMethod(Object proxy, Method method, Object[] args, String label) {
        switch (method.getName()) {
            case "toString":
                return label;
            case "hashCode":
                return System.identityHashCode(proxy);
            case "equals":
                return proxy == (args == null || args.length == 0 ? null : args[0]);
            default:
                return null;
        }
    }

    private Object invokeDefaultInterfaceMethod(Object proxy, Method method, Object[] args) throws Throwable {
        Class<?> declaringClass = method.getDeclaringClass();
        MethodHandles.Lookup lookup = MethodHandles.privateLookupIn(declaringClass, MethodHandles.lookup());
        return lookup.findSpecial(declaringClass,
                        method.getName(),
                        MethodType.methodType(method.getReturnType(), method.getParameterTypes()),
                        declaringClass)
                .bindTo(proxy)
                .invokeWithArguments(args == null ? new Object[0] : args);
    }

    private OutputStream resolveFlowOutputStreamHint(Map<String, Object> hints,
                                                     String methodName,
                                                     ClassLoader loader) throws Exception {
        Object raw = findFlowMethodHint(hints, methodName, "out");
        if (raw == null) {
            raw = findFlowMethodHint(hints, methodName, "return");
        }
        Object normalized = normalizeFlowNullLike(raw);
        if (normalized == null) {
            return new ByteArrayOutputStream();
        }
        if (normalized instanceof OutputStream) {
            return (OutputStream) normalized;
        }
        Object converted = convertValueOrTreatAsHint(normalized, OutputStream.class, loader);
        return converted instanceof OutputStream ? (OutputStream) converted : new ByteArrayOutputStream();
    }

    private InputStream resolveFlowInputStreamHint(Map<String, Object> hints,
                                                   String methodName,
                                                   ClassLoader loader) throws Exception {
        Object raw = findFlowMethodHint(hints, methodName, "returns");
        if (raw == null) {
            raw = findFlowMethodHint(hints, methodName, "return");
        }
        Object normalized = normalizeFlowNullLike(raw);
        if (normalized == null) {
            return new ByteArrayInputStream(new byte[]{1});
        }
        if (normalized instanceof InputStream) {
            return (InputStream) normalized;
        }
        Object converted = convertValueOrTreatAsHint(normalized, InputStream.class, loader);
        return converted instanceof InputStream ? (InputStream) converted : new ByteArrayInputStream(new byte[]{1});
    }

    private Object resolveFlowRecordHint(Object[] args, ClassLoader loader) throws Exception {
        Class<?> recordClass = Class.forName("com.lithium.flow.filer.Record", true, loader);
        String path = args != null && args.length > 0 && args[0] != null ? String.valueOf(args[0]) : "replay";
        Method noFile = recordClass.getMethod("noFile", URI.class, String.class);
        return noFile.invoke(null, URI.create("file:/"), path);
    }

    private Object findFlowMethodHint(Map<String, Object> hints, String methodName, String suffix) {
        if (hints == null) {
            return null;
        }
        String normalizedMethod = methodName == null ? "" : methodName.toLowerCase(Locale.ROOT);
        String normalizedSuffix = suffix == null ? "" : suffix.toLowerCase(Locale.ROOT);
        for (Map.Entry<String, Object> entry : hints.entrySet()) {
            String key = normalizeFlowHintKey(entry.getKey());
            if (key.contains(normalizedMethod) && key.contains(normalizedSuffix)) {
                return entry.getValue();
            }
        }
        return null;
    }

    private Object findFlowFunctionApplyHint(Map<String, Object> hints, Object arg) {
        if (hints == null) {
            return null;
        }
        String argText = arg == null ? "" : String.valueOf(arg);
        for (Map.Entry<String, Object> entry : hints.entrySet()) {
            String key = entry.getKey();
            if (key == null || !normalizeFlowHintKey(key).contains("apply")) {
                continue;
            }
            if (argText.isEmpty() || key.contains(argText)) {
                return entry.getValue();
            }
        }
        return null;
    }

    private boolean hasExplicitNullFunctionHint(Map<String, Object> hints, Object arg) {
        if (hints == null) {
            return false;
        }
        String argText = arg == null ? "" : String.valueOf(arg);
        for (Map.Entry<String, Object> entry : hints.entrySet()) {
            String key = entry.getKey();
            if (key == null || !normalizeFlowHintKey(key).contains("apply")) {
                continue;
            }
            if ((argText.isEmpty() || key.contains(argText)) && normalizeFlowNullLike(entry.getValue()) == null) {
                return true;
            }
        }
        return false;
    }

    private boolean flowHintRequestsNull(Map<String, Object> hints, String methodName) {
        if (hints == null) {
            return false;
        }
        for (Map.Entry<String, Object> entry : hints.entrySet()) {
            String key = normalizeFlowHintKey(entry.getKey());
            if (key.contains(methodName.toLowerCase(Locale.ROOT)) && key.contains("returnsnull")) {
                return true;
            }
        }
        return false;
    }

    private Object normalizeFlowNullLike(Object value) {
        if (!(value instanceof String)) {
            return value;
        }
        String raw = ((String) value).trim();
        if (raw.isEmpty()
                || "null".equalsIgnoreCase(raw)
                || "none".equalsIgnoreCase(raw)) {
            return null;
        }
        return value;
    }

    private Object defaultValue(Class<?> returnType) {
        if (returnType == null || returnType == Void.TYPE) {
            return null;
        }
        if (!returnType.isPrimitive()) {
            return null;
        }
        if (returnType == Boolean.TYPE) {
            return Boolean.FALSE;
        }
        if (returnType == Character.TYPE) {
            return '\0';
        }
        if (returnType == Byte.TYPE) {
            return (byte) 0;
        }
        if (returnType == Short.TYPE) {
            return (short) 0;
        }
        if (returnType == Integer.TYPE) {
            return 0;
        }
        if (returnType == Long.TYPE) {
            return 0L;
        }
        if (returnType == Float.TYPE) {
            return 0f;
        }
        if (returnType == Double.TYPE) {
            return 0d;
        }
        return null;
    }

    private void ensureParentDirectory(String path) {
        if (path == null || path.contains("://")) {
            return;
        }
        try {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null) {
                parent.mkdirs();
            }
        } catch (Throwable ignored) {
        }
    }

    private Map<String, Object> flattenImmediateFieldMap(AssignmentNode node) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (Map.Entry<String, AssignmentNode> entry : node.children.entrySet()) {
            if (entry.getValue().children.isEmpty()) {
                map.put(entry.getKey(), entry.getValue().value);
            }
        }
        return map;
    }

    private Map<String, Object> flattenLeafFieldMap(AssignmentNode node) {
        Map<String, Object> result = new LinkedHashMap<>();
        flattenLeafFieldMap(node, "", result);
        return result;
    }

    private void flattenLeafFieldMap(AssignmentNode node, String prefix, Map<String, Object> out) {
        if (node == null) {
            return;
        }
        if (node.children.isEmpty()) {
            if (!prefix.isEmpty()) {
                out.put(prefix, node.value);
            }
            return;
        }
        for (Map.Entry<String, AssignmentNode> entry : node.children.entrySet()) {
            String next = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            flattenLeafFieldMap(entry.getValue(), next, out);
        }
    }

    /**
     * When the current object does not define these fields, try routing them to a child object held by a Collection.
     * This is useful for structures such as:
     *   CompositeSelector
     *     -> List<KeyValueMapSelector> selectors
     *            -> PropertySelector(property, alias)
     */
    @SuppressWarnings("unchecked")
    private boolean routeUnmatchedToCollectionChildren(Object obj,
                                                       Map<String, AssignmentNode> unmatched,
                                                       ClassLoader loader,
                                                       String objectPath) throws Exception {
        List<Field> collectionFields = findCollectionFields(obj.getClass());
        if (collectionFields.isEmpty()) {
            return false;
        }

        boolean routed = false;

        for (Field collectionField : collectionFields) {
            collectionField.setAccessible(true);

            Collection<Object> collection = (Collection<Object>) collectionField.get(obj);
            if (collection == null) {
                collection = createMutableCollection(collectionField.getType());
                if (collection == null) {
                    continue;
                }
                collectionField.set(obj, collection);
            }

            Class<?> elementBaseType = resolveCollectionElementUpperBound(collectionField);
            if (elementBaseType == null || elementBaseType == Object.class) {
                elementBaseType = Object.class;
            }

            Class<?> bestCandidate = selectBestCollectionElementCandidate(obj.getClass(), elementBaseType, unmatched, loader);
            if (bestCandidate == null) {
                continue;
            }

            Object child = instantiateClass(bestCandidate, flattenUnmatchedLeafMap(unmatched), loader);
            AssignmentNode syntheticRoot = new AssignmentNode();
            syntheticRoot.children.putAll(unmatched);

            applyAssignmentNode(child, syntheticRoot, loader,
                    objectPath + "." + collectionField.getName() + "[" + bestCandidate.getSimpleName() + "]");

            collection.add(child);

            log.info("[DEBUG applyFieldValues] routed unmatched fields {} into collection field {}.{} using child type {}",
                    unmatched.keySet(),
                    obj.getClass().getName(),
                    collectionField.getName(),
                    bestCandidate.getName());

            routed = true;
            break;
        }

        return routed;
    }

    private Map<String, Object> flattenUnmatchedLeafMap(Map<String, AssignmentNode> unmatched) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (Map.Entry<String, AssignmentNode> e : unmatched.entrySet()) {
            if (e.getValue().children.isEmpty()) {
                result.put(e.getKey(), e.getValue().value);
            }
        }
        return result;
    }

    private List<Field> findCollectionFields(Class<?> clazz) {
        List<Field> result = new ArrayList<>();
        for (Field field : getAllFields(clazz)) {
            if (Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            if (Collection.class.isAssignableFrom(field.getType())) {
                result.add(field);
            }
        }
        return result;
    }

    private Class<?> resolveCollectionElementUpperBound(Field field) {
        try {
            Type genericType = field.getGenericType();
            if (genericType instanceof ParameterizedType) {
                Type[] actualTypeArgs = ((ParameterizedType) genericType).getActualTypeArguments();
                if (actualTypeArgs.length == 1) {
                    Type t = actualTypeArgs[0];
                    if (t instanceof Class<?>) {
                        return (Class<?>) t;
                    }
                    if (t instanceof ParameterizedType) {
                        Type raw = ((ParameterizedType) t).getRawType();
                        if (raw instanceof Class<?>) {
                            return (Class<?>) raw;
                        }
                    }
                }
            }
        } catch (Exception ignored) {
        }
        return Object.class;
    }

    private Class<?> resolveMapValueUpperBound(Field field, ClassLoader loader) {
        if (field == null) {
            return null;
        }
        try {
            Type genericType = field.getGenericType();
            if (genericType instanceof ParameterizedType) {
                Type[] actualTypeArgs = ((ParameterizedType) genericType).getActualTypeArguments();
                if (actualTypeArgs.length == 2) {
                    Type t = actualTypeArgs[1];
                    if (t instanceof Class<?>) {
                        return (Class<?>) t;
                    }
                    if (t instanceof ParameterizedType) {
                        Type raw = ((ParameterizedType) t).getRawType();
                        if (raw instanceof Class<?>) {
                            return (Class<?>) raw;
                        }
                    }
                }
            }
        } catch (Exception ignored) {
        }

        String fieldName = field.getName();
        if ("hFields".equals(fieldName)) {
            return resolveKnownMapValueType("org.apache.commons.validator.Field", loader);
        }
        if ("hActions".equals(fieldName)) {
            return resolveKnownMapValueType("org.apache.commons.validator.ValidatorAction", loader);
        }
        if ("hFormSets".equals(fieldName)) {
            return resolveKnownMapValueType("org.apache.commons.validator.FormSet", loader);
        }
        return null;
    }

    private Class<?> inferExistingMapValueType(Map<Object, Object> map) {
        if (map == null || map.isEmpty()) {
            return null;
        }

        Class<?> inferredType = null;
        for (Object value : map.values()) {
            if (value == null) {
                continue;
            }

            Class<?> valueType = value.getClass();
            if (inferredType == null) {
                inferredType = valueType;
                continue;
            }

            if (!inferredType.equals(valueType)) {
                return null;
            }
        }

        return inferredType;
    }

    private Class<?> selectBestCollectionElementCandidate(Class<?> ownerClass,
                                                      Class<?> elementBaseType,
                                                      Map<String, AssignmentNode> unmatched,
                                                      ClassLoader loader) {
        Set<Class<?>> candidates = discoverConcreteCandidates(ownerClass, elementBaseType, loader);

        int bestScore = Integer.MIN_VALUE;
        Class<?> best = null;

        for (Class<?> candidate : candidates) {
            int score = scoreCandidate(candidate, unmatched);
            if (score > bestScore) {
                bestScore = score;
                best = candidate;
            }
        }

        log.info("[DEBUG selectBestCollectionElementCandidate] ownerClass={}, elementBaseType={}, bestCandidate={}, bestScore={}, unmatchedKeys={}",
                ownerClass.getName(),
                elementBaseType == null ? "null" : elementBaseType.getName(),
                best == null ? "null" : best.getName(),
                bestScore,
                unmatched.keySet());

        if (best != null && bestScore > 0) {
            return best;
        }
        return null;
    }

    private int scoreCandidate(Class<?> candidate, Map<String, AssignmentNode> unmatched) {
        int score = 0;
    
        List<String> keys = new ArrayList<>(unmatched.keySet());
        Set<String> candidateFieldNames = new HashSet<>();
        List<Field> allFields = getAllFields(candidate);
    
        for (Field field : allFields) {
            candidateFieldNames.add(field.getName());
        }
    
        int directFieldHit = 0;
        for (String key : keys) {
            if (candidateFieldNames.contains(key)) {
                directFieldHit++;
                score += 20;   // Strong bonus for a direct field hit.
            }
        }
    
        // Constructors that can accept these fields also get a bonus, but with lower weight than direct field hits.
        Map<String, Object> leafMap = flattenUnmatchedLeafMap(unmatched);
        for (Constructor<?> constructor : candidate.getDeclaredConstructors()) {
            int local = 0;
            for (Parameter p : constructor.getParameters()) {
                String pName = p.isNamePresent() ? p.getName() : null;
                if (pName != null && leafMap.containsKey(pName)) {
                    local += 4;
                } else {
                    String inferred = inferFieldNameForParameter(
                            p,
                            allFields,
                            leafMap,
                            Collections.emptySet()
                    );
                    if (inferred != null) {
                        local += 3;
                    }
                }
            }
            score += local;
        }
    
        // Give a small bonus to concrete non-interface classes.
        if (!candidate.isInterface() && !Modifier.isAbstract(candidate.getModifiers())) {
            score += 2;
        }
    
        // A no-arg constructor makes stable instantiation easier.
        try {
            candidate.getDeclaredConstructor();
            score += 4;
        } catch (NoSuchMethodException ignored) {
        }
    
        // Fewer fields make the class look more like a leaf node, so add a small bonus.
        score += Math.max(0, 8 - allFields.size());
    
        // Holding another Collection suggests a container/composite node, so penalize it.
        for (Field field : allFields) {
            if (Collection.class.isAssignableFrom(field.getType())) {
                score -= 12;
            }
        }
    
        // Penalize class names that clearly indicate composite/container/nested roles.
        String simpleName = candidate.getSimpleName().toLowerCase(Locale.ROOT);
        if (simpleName.contains("composite")) {
            score -= 20;
        }
        if (simpleName.contains("nested")) {
            score -= 8;
        }
        if (simpleName.contains("classselector")) {
            score -= 10;
        }
    
        // Prefer leaf selectors such as property selectors.
        if (simpleName.contains("property")) {
            score += 15;
        }
    
        // Add another bonus when the unmatched fields are exactly property or alias.
        boolean wantsProperty = keys.contains("property");
        boolean wantsAlias = keys.contains("alias");
        if (wantsProperty && candidateFieldNames.contains("property")) {
            score += 12;
        }
        if (wantsAlias && candidateFieldNames.contains("alias")) {
            score += 8;
        }
    
        log.info("[DEBUG scoreCandidate] candidate={}, score={}, directFieldHit={}, keys={}",
                candidate.getName(), score, directFieldHit, keys);
    
        return score;
    }

    private Set<Class<?>> discoverConcreteCandidates(Class<?> ownerClass,
                                                     Class<?> baseType,
                                                     ClassLoader loader) {
        LinkedHashSet<Class<?>> result = new LinkedHashSet<>();

        if (baseType != null
                && baseType != Object.class
                && !baseType.isInterface()
                && !Modifier.isAbstract(baseType.getModifiers())) {
            result.add(baseType);
        }

        Set<String> packageNames = new LinkedHashSet<>();
        if (ownerClass.getPackage() != null) {
            packageNames.add(ownerClass.getPackage().getName());
        }
        if (baseType != null && baseType.getPackage() != null) {
            packageNames.add(baseType.getPackage().getName());
        }

        for (String pkg : packageNames) {
            for (Class<?> cls : scanConcreteClassesInPackage(pkg, loader)) {
                if (baseType == Object.class || baseType.isAssignableFrom(cls)) {
                    if (!cls.isInterface() && !Modifier.isAbstract(cls.getModifiers())) {
                        result.add(cls);
                    }
                }
            }
        }

        return result;
    }

    private Set<Class<?>> scanConcreteClassesInPackage(String packageName, ClassLoader loader) {
        LinkedHashSet<Class<?>> result = new LinkedHashSet<>();
        String packagePath = packageName.replace('.', '/');

        try {
            Enumeration<URL> resources = loader.getResources(packagePath);
            while (resources.hasMoreElements()) {
                URL url = resources.nextElement();
                if (!"file".equals(url.getProtocol())) {
                    continue;
                }

                String decodedPath = URLDecoder.decode(url.getPath(), StandardCharsets.UTF_8.name());
                File dir = new File(decodedPath);
                if (!dir.exists() || !dir.isDirectory()) {
                    continue;
                }

                File[] files = dir.listFiles();
                if (files == null) {
                    continue;
                }

                for (File file : files) {
                    String name = file.getName();
                    if (!name.endsWith(".class")) {
                        continue;
                    }
                    if (name.contains("$")) {
                        continue;
                    }

                    String simpleClassName = name.substring(0, name.length() - 6);
                    String fqcn = packageName + "." + simpleClassName;

                    try {
                        Class<?> cls = Class.forName(fqcn, false, loader);
                        if (!cls.isInterface() && !Modifier.isAbstract(cls.getModifiers())) {
                            result.add(cls);
                        }
                    } catch (Throwable ignored) {
                    }
                }
            }
        } catch (Exception ignored) {
        }

        return result;
    }

    @SuppressWarnings("unchecked")
    private Collection<Object> createMutableCollection(Class<?> fieldType) {
        if (fieldType == null) {
            return null;
        }
        if (isGuavaImmutableListType(fieldType)) {
            return new ArrayList<>();
        }
        if (isGuavaImmutableSetType(fieldType)) {
            return new LinkedHashSet<>();
        }
        if (!fieldType.isInterface() && !Modifier.isAbstract(fieldType.getModifiers())) {
            try {
                Constructor<?> c = fieldType.getDeclaredConstructor();
                c.setAccessible(true);
                return (Collection<Object>) c.newInstance();
            } catch (Exception ignored) {
            }
        }
        if (BlockingQueue.class.isAssignableFrom(fieldType)) {
            return new LinkedBlockingQueue<>();
        }
        if (Deque.class.isAssignableFrom(fieldType)) {
            return new ArrayDeque<>();
        }
        if (Queue.class.isAssignableFrom(fieldType)) {
            return new LinkedList<>();
        }
        if (List.class.isAssignableFrom(fieldType) || Collection.class.equals(fieldType)) {
            return new LinkedList<>();
        }
        if (Set.class.isAssignableFrom(fieldType)) {
            return new LinkedHashSet<>();
        }
        return new LinkedList<>();
    }

    private boolean isGuavaImmutableListType(Class<?> fieldType) {
        return fieldType != null
                && "com.google.common.collect.ImmutableList".equals(fieldType.getName());
    }

    private boolean isGuavaImmutableSetType(Class<?> fieldType) {
        return fieldType != null
                && "com.google.common.collect.ImmutableSet".equals(fieldType.getName());
    }

    private Object finalizeCollectionForFieldType(Class<?> fieldType,
                                                  Collection<Object> collection,
                                                  ClassLoader loader) {
        if (collection == null || fieldType == null) {
            return collection;
        }

        try {
            if (isGuavaImmutableListType(fieldType)) {
                Method copyOf = fieldType.getMethod("copyOf", Iterable.class);
                return copyOf.invoke(null, collection);
            }
            if (isGuavaImmutableSetType(fieldType)) {
                Method copyOf = fieldType.getMethod("copyOf", Iterable.class);
                return copyOf.invoke(null, collection);
            }
        } catch (Throwable e) {
            log.info("[DEBUG finalizeCollectionForFieldType] failed to finalize {}: {}",
                    fieldType.getName(), e.toString());
        }

        return collection;
    }

    private void populateCollectionFromHint(Collection<Object> collection,
                                            Class<?> elementType,
                                            Object rawValue,
                                            ClassLoader loader) throws Exception {
        if (collection == null || rawValue == null) {
            return;
        }

        if (!(rawValue instanceof String)) {
            Object element = elementType == null
                    ? rawValue
                    : convertValueOrTreatAsHint(rawValue, elementType, loader);
            if (element != null) {
                collection.add(element);
            }
            return;
        }

        String raw = ((String) rawValue).trim();
        if (raw.isEmpty()) {
            return;
        }

        if (elementType != null
                && "com.lithium.flow.streams.Streamer".equals(elementType.getName())
                && raw.toLowerCase(Locale.ROOT).contains("non-null streamer")) {
            collection.add(createFlowStreamerProxy(elementType, Collections.emptyMap(), loader));
        }

        for (String handle : extractHandleSequence(raw)) {
            String hintedClassName = extractClassHint(handle);
            if (hintedClassName != null && elementType != null) {
                try {
                    Class<?> hintedClass = Class.forName(hintedClassName, true, loader);
                    if (Collection.class.isAssignableFrom(hintedClass)
                            && !Collection.class.isAssignableFrom(elementType)) {
                        continue;
                    }
                } catch (Throwable ignored) {
                }
            }
            Object element = rebuildObjectFromHandle(handle,
                    elementType == null ? Object.class : elementType,
                    currentFieldValueContext == null ? Collections.emptyMap() : currentFieldValueContext,
                    loader);
            if (element != null) {
                collection.add(element);
            }
        }
    }

    private List<String> extractHandleSequence(Object rawValue) {
        List<String> handles = new ArrayList<>();
        if (!(rawValue instanceof String)) {
            return handles;
        }
        Matcher matcher = HANDLE_PATTERN.matcher(((String) rawValue).trim());
        while (matcher.find()) {
            handles.add(matcher.group(1));
        }
        return handles;
    }

    private Object convertValueOrTreatAsHint(Object rawValue, Class<?> targetType, ClassLoader loader) throws Exception {
        if (rawValue == null) {
            return convertValue(null, targetType, loader);
        }

        if (rawValue instanceof String && targetType != String.class) {
            if (isSymbolicHandle((String) rawValue)) {
                Object rebuilt = rebuildObjectFromHandle((String) rawValue,
                        targetType,
                        currentFieldValueContext == null ? Collections.emptyMap() : currentFieldValueContext,
                        loader);
                if (rebuilt != null) {
                    return rebuilt;
                }
            }

            String hint = extractClassHint((String) rawValue);
            if (hint != null) {
                Class<?> hintedClass;
                try {
                    hintedClass = Class.forName(hint, true, loader);
                } catch (Throwable e) {
                    hintedClass = null;
                }
                if (hintedClass != null && targetType.isAssignableFrom(hintedClass)) {
                    return instantiateClass(hintedClass, Collections.emptyMap(), loader);
                }
            }
        }

        return convertValue(rawValue, targetType, loader);
    }

    private Object convertValue(Object rawValue, Class<?> targetType, ClassLoader loader) throws Exception {
        if (rawValue == null) {
            if (targetType.isPrimitive()) {
                return primitiveDefaultValue(targetType);
            }
            return null;
        }

        if (targetType == String.class) {
            return String.valueOf(rawValue);
        }
        if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(String.valueOf(rawValue));
        }
        if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(String.valueOf(rawValue));
        }
        if (targetType == boolean.class || targetType == Boolean.class) {
            return Boolean.parseBoolean(String.valueOf(rawValue));
        }
        if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(String.valueOf(rawValue));
        }
        if (targetType == float.class || targetType == Float.class) {
            return Float.parseFloat(String.valueOf(rawValue));
        }
        if (targetType == short.class || targetType == Short.class) {
            return Short.parseShort(String.valueOf(rawValue));
        }
        if (targetType == byte.class || targetType == Byte.class) {
            return Byte.parseByte(String.valueOf(rawValue));
        }
        if (targetType == char.class || targetType == Character.class) {
            String s = String.valueOf(rawValue);
            return s.isEmpty() ? '\0' : s.charAt(0);
        }
        if (targetType == URI.class) {
            String s = String.valueOf(rawValue).trim();
            return s.isEmpty() || "null".equalsIgnoreCase(s) ? null : URI.create(s);
        }
        if (targetType == Locale.class) {
            return convertToLocale(rawValue);
        }
        if ("org.apache.http.HttpHost".equals(targetType.getName())) {
            return convertToHttpHost(rawValue, targetType);
        }
        if (targetType == Optional.class) {
            return convertToOptional(rawValue);
        }

        if (Map.class.isAssignableFrom(targetType) && rawValue instanceof Map) {
            return normalizeMapEntries((Map<?, ?>) rawValue, loader);
        }

        if (Collection.class.isAssignableFrom(targetType) && rawValue instanceof List) {
            return normalizeListElements((List<?>) rawValue);
        }

        if (targetType.isInstance(rawValue)) {
            return rawValue;
        }

        if (targetType.isArray()) {
            if (rawValue instanceof List) {
                List<?> list = (List<?>) rawValue;
                Class<?> componentType = targetType.getComponentType();
                Object array = Array.newInstance(componentType, list.size());
                for (int i = 0; i < list.size(); i++) {
                    Array.set(array, i, convertValue(list.get(i), componentType, loader));
                }
                return array;
            }
            if (rawValue instanceof String) {
                try {
                    List<?> parsed = objectMapper.readValue((String) rawValue, List.class);
                    Class<?> componentType = targetType.getComponentType();
                    Object array = Array.newInstance(componentType, parsed.size());
                    for (int i = 0; i < parsed.size(); i++) {
                        Array.set(array, i, convertValue(parsed.get(i), componentType, loader));
                    }
                    return array;
                } catch (Exception ignored) {
                }
                List<?> looseParsed = parseLooseListString((String) rawValue);
                if (looseParsed != null) {
                    Class<?> componentType = targetType.getComponentType();
                    Object array = Array.newInstance(componentType, looseParsed.size());
                    for (int i = 0; i < looseParsed.size(); i++) {
                        Array.set(array, i, convertValue(looseParsed.get(i), componentType, loader));
                    }
                    return array;
                }
            }
        }

        if (Collection.class.isAssignableFrom(targetType)) {
            if (rawValue instanceof String) {
                try {
                    List<?> parsed = objectMapper.readValue((String) rawValue, List.class);
                    return normalizeListElements(parsed);
                } catch (Exception ignored) {
                }
                List<?> looseParsed = parseLooseListString((String) rawValue);
                if (looseParsed != null) {
                    return normalizeListElements(looseParsed);
                }
            }
        }

        if (rawValue instanceof String) {
            String rawString = (String) rawValue;
            try {
                if (rawString.startsWith("[") || rawString.startsWith("{")) {
                    Object parsed = objectMapper.readValue(rawString, Object.class);
                    return objectMapper.convertValue(parsed, targetType);
                }
            } catch (Exception ignored) {
            }
        }

        return objectMapper.convertValue(rawValue, targetType);
    }

    private Optional<?> convertToOptional(Object rawValue) {
        if (rawValue == null) {
            return Optional.empty();
        }
        if (rawValue instanceof Optional) {
            return (Optional<?>) rawValue;
        }
        if (rawValue instanceof String) {
            String text = ((String) rawValue).trim();
            if (text.isEmpty() || "null".equalsIgnoreCase(text)) {
                return Optional.empty();
            }
            if (isSymbolicHandle(text) && text.startsWith("java.util.Optional@")) {
                return Optional.empty();
            }
            return Optional.of(text);
        }
        return Optional.of(rawValue);
    }

    private Object convertToHttpHost(Object rawValue, Class<?> targetType) throws Exception {
        if (rawValue == null) {
            return null;
        }
        if (targetType.isInstance(rawValue)) {
            return rawValue;
        }

        String s = String.valueOf(rawValue).trim();
        if (s.isEmpty() || "null".equalsIgnoreCase(s)) {
            return null;
        }

        try {
            Method create = targetType.getDeclaredMethod("create", String.class);
            create.setAccessible(true);
            return create.invoke(null, s);
        } catch (NoSuchMethodException ignored) {
        }

        try {
            Constructor<?> ctor = targetType.getDeclaredConstructor(String.class);
            ctor.setAccessible(true);
            return ctor.newInstance(s);
        } catch (NoSuchMethodException ignored) {
        }

        return objectMapper.convertValue(rawValue, targetType);
    }

    private Locale convertToLocale(Object rawValue) {
        if (rawValue instanceof Locale) {
            return (Locale) rawValue;
        }

        String s = String.valueOf(rawValue).trim();
        if (s.isEmpty() || "null".equalsIgnoreCase(s)) {
            return null;
        }

        if (s.startsWith("java.util.Locale@")) {
            s = s.substring("java.util.Locale@".length()).trim();
        } else if (s.startsWith("Locale@")) {
            s = s.substring("Locale@".length()).trim();
        }

        if (s.contains("_")) {
            String[] parts = s.split("_", -1);
            if (parts.length == 1) {
                return new Locale(parts[0]);
            }
            if (parts.length == 2) {
                return new Locale(parts[0], parts[1]);
            }
            return new Locale(parts[0], parts[1], parts[2]);
        }

        if (s.contains("-")) {
            return Locale.forLanguageTag(s);
        }

        return new Locale(s);
    }

    private Map<Object, Object> normalizeMapEntries(Map<?, ?> rawMap, ClassLoader loader) throws Exception {
        Map<Object, Object> normalized = new LinkedHashMap<>();
        if (rawMap == null) {
            return normalized;
        }

        for (Map.Entry<?, ?> entry : rawMap.entrySet()) {
            Object rawKey = entry.getKey();
            String canonicalKey = canonicalizeWellKnownLiteral(rawKey == null ? null : String.valueOf(rawKey));
            Object normalizedKey = canonicalKey == null ? rawKey : canonicalKey;
            Object normalizedValue = adaptKnownMapValue(canonicalKey, entry.getValue(), loader);
            normalized.put(normalizedKey, normalizedValue);
        }

        return normalized;
    }

    private Object adaptKnownMapValue(String canonicalKey, Object rawValue, ClassLoader loader) throws Exception {
        Class<?> targetType = resolveKnownMapValueType(canonicalKey, loader);
        if (targetType == null) {
            return rawValue;
        }

        if (targetType == Locale.class) {
            return convertToLocale(rawValue);
        }

        if (rawValue instanceof String && isSymbolicHandle((String) rawValue)) {
            Object rebuilt = rebuildObjectFromHandle((String) rawValue, targetType, Collections.emptyMap(), loader);
            if (rebuilt != null) {
                return rebuilt;
            }
        }

        if (targetType == Object.class) {
            if (rawValue instanceof String && isSymbolicHandle((String) rawValue)) {
                Object rebuilt = rebuildObjectFromHandle((String) rawValue, Object.class, Collections.emptyMap(), loader);
                if (rebuilt != null) {
                    return rebuilt;
                }
            }
            return rawValue;
        }

        return convertValue(rawValue, targetType, loader);
    }

    private Class<?> resolveKnownMapValueType(String canonicalKey, ClassLoader loader) {
        if (canonicalKey == null) {
            return null;
        }
        switch (canonicalKey) {
            case "java.lang.Object":
                return Object.class;
            case "java.util.Locale":
                return Locale.class;
            case "org.apache.commons.validator.Field":
            case "org.apache.commons.validator.ValidatorAction":
            case "org.apache.commons.validator.ValidatorResults":
            case "org.apache.commons.validator.Form":
            case "org.apache.commons.validator.Validator":
                try {
                    return Class.forName(canonicalKey, true, loader);
                } catch (Throwable ignored) {
                    return null;
                }
            default:
                return null;
        }
    }

    private List<Object> normalizeListElements(List<?> input) {
        List<Object> normalized = new ArrayList<>();
        if (input == null) {
            return normalized;
        }

        for (Object item : input) {
            if (item instanceof String) {
                normalized.add(canonicalizeWellKnownLiteral((String) item));
            } else {
                normalized.add(item);
            }
        }
        return normalized;
    }

    private List<?> parseLooseListString(String rawValue) {
        if (rawValue == null) {
            return null;
        }

        String s = rawValue.trim();
        if (!s.endsWith("]")) {
            return null;
        }
        if (!s.startsWith("[")) {
            int containsPos = s.indexOf("contains=[");
            if (containsPos >= 0) {
                s = s.substring(s.indexOf('[', containsPos));
            } else {
                int firstBracket = s.indexOf('[');
                if (firstBracket < 0) {
                    return null;
                }
                s = s.substring(firstBracket);
            }
        }

        if (!s.startsWith("[") || !s.endsWith("]")) {
            return null;
        }

        String body = s.substring(1, s.length() - 1).trim();
        if (body.isEmpty()) {
            return Collections.emptyList();
        }

        Matcher repeatedArraySummary = Pattern
                .compile("(-?\\d+)\\s*x\\s*(\\d+)\\s*elements?", Pattern.CASE_INSENSITIVE)
                .matcher(body);
        if (repeatedArraySummary.matches()) {
            String repeatedValue = repeatedArraySummary.group(1);
            long requestedLength = Long.parseLong(repeatedArraySummary.group(2));
            int replayLength = (int) Math.min(requestedLength, 16L);
            List<String> repeatedValues = new ArrayList<>();
            for (int i = 0; i < replayLength; i++) {
                repeatedValues.add(repeatedValue);
            }
            return repeatedValues;
        }

        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        int quote = 0;

        for (int i = 0; i < body.length(); i++) {
            char ch = body.charAt(i);
            if (ch == '"' || ch == '\'') {
                quote = quote == 0 ? ch : (quote == ch ? 0 : quote);
                continue;
            }

            if (ch == ',' && quote == 0) {
                values.add(current.toString().trim());
                current.setLength(0);
                continue;
            }
            current.append(ch);
        }
        values.add(current.toString().trim());
        return values;
    }

    private Object primitiveDefaultValue(Class<?> type) {
        if (type == boolean.class) return false;
        if (type == byte.class) return (byte) 0;
        if (type == short.class) return (short) 0;
        if (type == int.class) return 0;
        if (type == long.class) return 0L;
        if (type == float.class) return 0f;
        if (type == double.class) return 0d;
        if (type == char.class) return '\0';
        return null;
    }

    private Class<?> resolveType(String typeName, ClassLoader loader) throws ClassNotFoundException {
        switch (typeName) {
            case "boolean": return boolean.class;
            case "byte": return byte.class;
            case "short": return short.class;
            case "int": return int.class;
            case "long": return long.class;
            case "float": return float.class;
            case "double": return double.class;
            case "char": return char.class;
            case "void": return void.class;
            default:
                if (typeName.endsWith("[]")) {
                    Class<?> componentType = resolveType(typeName.substring(0, typeName.length() - 2), loader);
                    return Array.newInstance(componentType, 0).getClass();
                }
                return Class.forName(typeName, true, loader);
        }
    }

    private MethodDescriptor parseMethodSignature(String signature) {
        String sig = signature.trim();

        int colonIndex = sig.indexOf(':');
        int leftParenIndex = sig.indexOf('(');
        int rightParenIndex = sig.lastIndexOf(')');

        if (colonIndex < 0 || leftParenIndex < 0 || rightParenIndex < 0) {
            throw new RuntimeException("Unsupported method signature: " + signature);
        }

        String left = sig.substring(0, colonIndex).trim();
        String right = sig.substring(colonIndex + 1).trim();

        int lastDot = left.lastIndexOf('.');
        if (lastDot < 0) {
            throw new RuntimeException("Invalid method signature, missing class name: " + signature);
        }

        String className = left.substring(0, lastDot).trim();
        String methodName = left.substring(lastDot + 1).trim();
        String returnType = right.substring(0, right.indexOf('(')).trim();

        String paramsRaw = right.substring(right.indexOf('(') + 1, right.lastIndexOf(')')).trim();
        List<String> paramTypes;
        if (paramsRaw.isEmpty()) {
            paramTypes = Collections.emptyList();
        } else {
            paramTypes = new ArrayList<>();
            for (String p : paramsRaw.split(",")) {
                String v = p.trim();
                if (!v.isEmpty()) {
                    paramTypes.add(v);
                }
            }
        }

        MethodDescriptor descriptor = new MethodDescriptor();
        descriptor.setRawSignature(signature);
        descriptor.setClassName(className);
        descriptor.setMethodName(methodName);
        descriptor.setReturnTypeName(returnType);
        descriptor.setParamTypeNames(paramTypes);
        return descriptor;
    }

    @Data
    public static class ReplayResult {
        private boolean entryInvocationSuccess;
        private Object entryReturnValue;
        private Throwable entryThrowValue;
        private boolean sinkReached;
        private String resultStatus;
        private InvokeMethodResult invokeMethodResult;
        private Object replayTarget;
        private Object[] entryArgs;
    }

    @Data
    public static class ValidationInput {
        private String entryMethod;
        private String sinkMethod;
        private FinalPayload finalPayload;
    }

    @Data
    public static class FinalPayload {
        private Map<String, Object> argValues = Collections.emptyMap();
        private Map<String, Object> fieldValues = Collections.emptyMap();
    }

    @Data
    private static class MethodDescriptor {
        private String rawSignature;
        private String className;
        private String methodName;
        private String returnTypeName;
        private List<String> paramTypeNames = Collections.emptyList();

        boolean isConstructor() {
            return "<init>".equals(methodName);
        }
    }

    @Data
    private static class ConstructorBuildResult {
        private boolean matched;
        private Object[] args = new Object[0];
        private Set<String> consumedFieldNames = Collections.emptySet();

        static ConstructorBuildResult notMatched() {
            ConstructorBuildResult result = new ConstructorBuildResult();
            result.setMatched(false);
            return result;
        }
    }

    private static class AssignmentNode {
        private Object value;
        private final Map<String, AssignmentNode> children = new LinkedHashMap<>();
    }

    private static class SimpleHttpExchange extends HttpExchange {
        private Headers requestHeaders = new Headers();
        private Headers responseHeaders = new Headers();
        private URI requestURI = URI.create("/v1/test");
        private String requestMethod = "POST";
        private String requestBody = "";
        private ByteArrayOutputStream responseBody = new ByteArrayOutputStream();
        private Map<String, Object> attributes = new LinkedHashMap<>();
        private int responseCode = -1;

        @Override
        public Headers getRequestHeaders() {
            return requestHeaders;
        }

        @Override
        public Headers getResponseHeaders() {
            return responseHeaders;
        }

        @Override
        public URI getRequestURI() {
            return requestURI;
        }

        @Override
        public String getRequestMethod() {
            return requestMethod;
        }

        @Override
        public com.sun.net.httpserver.HttpContext getHttpContext() {
            return null;
        }

        @Override
        public void close() {
        }

        @Override
        public InputStream getRequestBody() {
            return new ByteArrayInputStream(requestBody == null
                    ? new byte[0]
                    : requestBody.getBytes(StandardCharsets.UTF_8));
        }

        @Override
        public OutputStream getResponseBody() {
            return responseBody;
        }

        @Override
        public void sendResponseHeaders(int rCode, long responseLength) {
            this.responseCode = rCode;
        }

        @Override
        public InetSocketAddress getRemoteAddress() {
            return new InetSocketAddress("127.0.0.1", 0);
        }

        @Override
        public int getResponseCode() {
            return responseCode;
        }

        @Override
        public InetSocketAddress getLocalAddress() {
            return new InetSocketAddress("127.0.0.1", 0);
        }

        @Override
        public String getProtocol() {
            return "HTTP/1.1";
        }

        @Override
        public Object getAttribute(String name) {
            return attributes.get(name);
        }

        @Override
        public void setAttribute(String name, Object value) {
            attributes.put(name, value);
        }

        @Override
        public void setStreams(InputStream i, OutputStream o) {
            if (o instanceof ByteArrayOutputStream) {
                responseBody = (ByteArrayOutputStream) o;
            }
        }

        @Override
        public HttpPrincipal getPrincipal() {
            return null;
        }
    }

    private static class StaticFieldTarget {
        private Class<?> ownerClass;
        private String fieldName;
    }

    // private void applyMethodLikeHints(Object target,
    //                               Map<String, Object> fieldValues,
    //                               ClassLoader loader) throws Exception {
    //     if (target == null || fieldValues == null || fieldValues.isEmpty()) {
    //         return;
    //     }

    //     for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
    //         String rawKey = entry.getKey();
    //         Object rawValue = entry.getValue();

    //         // 1) Handle this.xxx.getHeaders("Cookie").
    //         if (rawKey.startsWith("this.") && rawKey.contains(".getHeaders(")) {
    //             applyGetHeadersHint(target, rawKey, rawValue, loader);
    //             continue;
    //         }

    //         // 2) Handle headers(COOKIE_HEADER).
    //         if (rawKey.startsWith("headers(")) {
    //             applyHeadersMethodHint(target, rawKey, rawValue, fieldValues, loader);
    //             continue;
    //         }

    //         // 3) Handle headerValues[0].
    //         if (rawKey.startsWith("headerValues[")) {
    //             applyHeaderValuesAliasHint(target, rawKey, rawValue, fieldValues, loader);
    //         }
    //     }
    // }

    private void applyMethodLikeHints(Object target,
                                  Map<String, Object> fieldValues,
                                  ClassLoader loader) throws Exception {
        if (target == null || fieldValues == null || fieldValues.isEmpty()) {
            return;
        }

        // Apply hints to the current object first.
        applyMethodLikeHintsToSingleObject(target, fieldValues, loader);

        // Then apply them recursively to common delegate fields.
        applyMethodLikeHintsRecursively(target, fieldValues, loader, new HashSet<>());
    }

    private void applyMethodLikeHintsRecursively(Object target,
                                             Map<String, Object> fieldValues,
                                             ClassLoader loader,
                                             Set<Object> visited) throws Exception {
        if (target == null || visited.contains(target)) {
            return;
        }
        visited.add(target);

        for (String delegateFieldName : Arrays.asList("request", "httpRequest")) {
            Field f = tryFindField(target.getClass(), delegateFieldName);
            if (f == null) {
                continue;
            }

            f.setAccessible(true);
            Object nested = f.get(target);
            if (nested == null) {
                continue;
            }

            Map<String, Object> nestedFieldValues = remapFieldValuesForDelegate(fieldValues, delegateFieldName);
            if (!nestedFieldValues.isEmpty()) {
                applyMethodLikeHintsToSingleObject(nested, nestedFieldValues, loader);
            }

            applyMethodLikeHintsRecursively(nested, nestedFieldValues, loader, visited);
        }
    }

    private Map<String, Object> remapFieldValuesForDelegate(Map<String, Object> original,
                                                        String delegateFieldName) {
        Map<String, Object> remapped = new LinkedHashMap<>();
        String prefix = "this." + delegateFieldName + ".";

        for (Map.Entry<String, Object> entry : original.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            // 1) this.request.xxx -> this.xxx from the child's perspective.
            if (key.startsWith(prefix)) {
                remapped.put("this." + key.substring(prefix.length()), value);
                continue;
            }

            // 2) Do not forward the object handle itself, such as this.request; it is a parent field, not a child field.
            if (key.equals("this." + delegateFieldName)) {
                continue;
            }

            // 3) Generic hints: forward them directly to the child.
            if (key.startsWith("headers(")
                    || key.equals("headers")
                    || key.startsWith("headers[")
                    || key.equals("headerValues")
                    || key.startsWith("headerValues[")
                    || key.equals("COOKIE_HEADER")
                    || key.startsWith("CookieDecoder.decode(")) {
                remapped.put(key, value);
                continue;
            }

            // 4) Also forward common child-local field and path hints.
            if (key.startsWith("this.httpRequest.")
                    || key.startsWith("this.headers.")
                    || key.startsWith("this.headers[")
                    || key.equals("this.headers")
                    || key.equals("this.COOKIE_HEADER")
                    || key.startsWith("this.request.")) {
                remapped.put(key, value);
            }
        }

        return remapped;
    }

    private void applyMethodLikeHintsToSingleObject(Object target,
                                                Map<String, Object> fieldValues,
                                                ClassLoader loader) throws Exception {
        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String rawKey = entry.getKey();
            Object rawValue = entry.getValue();

            if (rawKey.startsWith("this.") && rawKey.contains(".getHeaders(")) {
                applyGetHeadersHint(target, rawKey, rawValue, loader);
                continue;
            }

            if (rawKey.startsWith("this.") && rawKey.contains(".headers[")) {
                applyBracketHeaderStorageHint(target, rawKey, rawValue, loader);
                continue;
            }

            if (rawKey.startsWith("this.") && rawKey.contains(".headers.")) {
                applyDotHeaderStorageHint(target, rawKey, rawValue, loader);
                continue;
            }

            // Also support direct fields such as this.headers and this.COOKIE_HEADER.
            if (rawKey.equals("this.headers") || rawKey.startsWith("this.headers[")) {
                applyThisDirectHeadersFieldHint(target, rawKey, rawValue, fieldValues, loader);
                continue;
            }

            if (rawKey.equals("this.COOKIE_HEADER")) {
                applyThisCookieHeaderConstantHint(target, rawKey, rawValue, loader);
                continue;
            }

            if (rawKey.startsWith("headers(")) {
                applyHeadersMethodHint(target, rawKey, rawValue, fieldValues, loader);
                continue;
            }

            if (rawKey.equals("headers") || rawKey.startsWith("headers[")) {
                applyDirectHeadersFieldHint(target, rawKey, rawValue, fieldValues, loader);
                continue;
            }

            if (rawKey.startsWith("headerValues[") || rawKey.equals("headerValues")) {
                applyHeaderValuesAliasHint(target, rawKey, rawValue, fieldValues, loader);
            }
        }
    }

    private void applyThisDirectHeadersFieldHint(Object root,
                                             String rawKey,
                                             Object rawValue,
                                             Map<String, Object> fieldValues,
                                             ClassLoader loader) throws Exception {
        String headerName = "Cookie";

        Object cookieHeaderAlias = fieldValues.get("this.COOKIE_HEADER");
        if (cookieHeaderAlias == null) {
            cookieHeaderAlias = fieldValues.get("COOKIE_HEADER");
        }
        if (cookieHeaderAlias != null) {
            headerName = String.valueOf(cookieHeaderAlias);
        }

        List<String> values = new ArrayList<>();

        if (rawKey.equals("this.headers")) {
            SortedMap<Integer, String> indexed = collectIndexedStringValues("headerValues", fieldValues);
            if (!indexed.isEmpty()) {
                values.addAll(indexed.values());
            } else if (rawValue != null) {
                values.add(String.valueOf(rawValue));
            }
        } else if (rawKey.startsWith("this.headers[")) {
            values = toStringList(rawValue);
        }

        if (values.isEmpty()) {
            return;
        }

        boolean applied = tryApplyHeaderValues(root, headerName, values);
        log.info("[DEBUG applyThisDirectHeadersFieldHint] root={}, headerName={}, values={}, applied={}",
                root.getClass().getName(), headerName, values, applied);
    }

    private void applyThisCookieHeaderConstantHint(Object root,
                                               String rawKey,
                                               Object rawValue,
                                               ClassLoader loader) {
        Field f = tryFindField(root.getClass(), "COOKIE_HEADER");
        if (f == null) {
            return;
        }

        try {
            f.setAccessible(true);

            // Avoid forcing static final constants; only try when the field is writable.
            int mod = f.getModifiers();
            if (Modifier.isFinal(mod)) {
                log.info("[DEBUG applyThisCookieHeaderConstantHint] skip final field {}.COOKIE_HEADER",
                        root.getClass().getName());
                return;
            }

            f.set(root, String.valueOf(rawValue));
            log.info("[DEBUG applyThisCookieHeaderConstantHint] set {}.COOKIE_HEADER={}",
                    root.getClass().getName(), rawValue);
        } catch (Exception e) {
            log.info("[DEBUG applyThisCookieHeaderConstantHint] failed to set {}.COOKIE_HEADER: {}",
                    root.getClass().getName(), e.toString());
        }
    }


    private void applyDirectHeadersFieldHint(Object root,
                                         String rawKey,
                                         Object rawValue,
                                         Map<String, Object> fieldValues,
                                         ClassLoader loader) throws Exception {
        String headerName = "Cookie";
        Object cookieHeaderAlias = fieldValues.get("COOKIE_HEADER");
        if (cookieHeaderAlias != null) {
            headerName = String.valueOf(cookieHeaderAlias);
        }

        List<String> values = new ArrayList<>();

        if (rawKey.equals("headers")) {
            // rawValue is often only a stringified printout here, so prefer values collected from headerValues[...].
            SortedMap<Integer, String> indexed = collectIndexedStringValues("headerValues", fieldValues);
            if (!indexed.isEmpty()) {
                values.addAll(indexed.values());
            } else if (rawValue != null) {
                values.add(String.valueOf(rawValue));
            }
        } else if (rawKey.startsWith("headers[")) {
            values = toStringList(rawValue);
        }

        if (values.isEmpty()) {
            return;
        }

        boolean applied = tryApplyHeaderValues(root, headerName, values);
        log.info("[DEBUG applyDirectHeadersFieldHint] root={}, headerName={}, values={}, applied={}",
                root.getClass().getName(), headerName, values, applied);
    }

    private void applyBracketHeaderStorageHint(Object root,
                                           String rawKey,
                                           Object rawValue,
                                           ClassLoader loader) throws Exception {
        // Supports:
        // this.httpRequest.headers["Cookie"][0]
        // this.httpRequest.headers["Cookie"]
        int headersPos = rawKey.indexOf(".headers[");
        if (headersPos < 0) {
            return;
        }

        String objectPath = rawKey.substring(0, headersPos); // this.httpRequest
        Object receiver = resolveObjectPath(root, objectPath);
        if (receiver == null) {
            return;
        }

        int firstQuote = rawKey.indexOf('"');
        int secondQuote = rawKey.indexOf('"', firstQuote + 1);
        if (firstQuote < 0 || secondQuote < 0) {
            return;
        }

        String headerName = rawKey.substring(firstQuote + 1, secondQuote);
        List<String> values = toStringList(rawValue);
        if (values.isEmpty()) {
            return;
        }

        boolean applied = tryApplyHeaderValues(receiver, headerName, values);
        log.info("[DEBUG applyBracketHeaderStorageHint] receiver={}, headerName={}, values={}, applied={}",
                receiver.getClass().getName(), headerName, values, applied);
    }
    private String rewritePrefixToThis(String rawKey, String prefix) {
        if (rawKey.startsWith(prefix + ".")) {
            return "this." + rawKey.substring((prefix + ".").length());
        }
        return rawKey;
    }

    private void applyDotHeaderStorageHint(Object root,
                                       String rawKey,
                                       Object rawValue,
                                       ClassLoader loader) throws Exception {
        // Supports:
        // this.request.headers.Cookie[0]
        int headersPos = rawKey.indexOf(".headers.");
        if (headersPos < 0) {
            return;
        }

        String objectPath = rawKey.substring(0, headersPos); // this.request
        Object receiver = resolveObjectPath(root, objectPath);
        if (receiver == null) {
            return;
        }

        String tail = rawKey.substring(headersPos + ".headers.".length()); // Cookie[0]
        String headerName = tail;
        int bracketPos = tail.indexOf('[');
        if (bracketPos >= 0) {
            headerName = tail.substring(0, bracketPos);
        }

        List<String> values = toStringList(rawValue);
        if (values.isEmpty()) {
            return;
        }

        boolean applied = tryApplyHeaderValues(receiver, headerName, values);
        log.info("[DEBUG applyDotHeaderStorageHint] receiver={}, headerName={}, values={}, applied={}",
                receiver.getClass().getName(), headerName, values, applied);
    }



    private void applyGetHeadersHint(Object root,
                                 String rawKey,
                                 Object rawValue,
                                 ClassLoader loader) throws Exception {
        // Example:
        // this.httpRequest.getHeaders("Cookie")
        int methodPos = rawKey.indexOf(".getHeaders(");
        String objectPath = rawKey.substring(0, methodPos); // this.httpRequest
        String methodArg = rawKey.substring(rawKey.indexOf('(') + 1, rawKey.lastIndexOf(')')).trim();

        String headerName = stripQuotes(methodArg);
        List<String> values = toStringList(rawValue);
        if (values.isEmpty()) {
            return;
        }

        Object receiver = resolveObjectPath(root, objectPath);

        // 1) First try the original real-object state injection path.
        if (receiver != null) {
            boolean applied = tryApplyHeaderValues(receiver, headerName, values);
            if (applied) {
                log.info("[DEBUG applyGetHeadersHint] applied getHeaders hint to receiver={}, headerName={}, values={}",
                        receiver.getClass().getName(), headerName, values);
                return;
            }
        }

        // 2) If real-object injection fails, fall back to replacing the field with a method-return stub/proxy.
        boolean stubbed = installGetHeadersStub(root, objectPath, headerName, values, loader);
        if (stubbed) {
            log.info("[DEBUG applyGetHeadersHint] installed getHeaders stub for path={}, headerName={}, values={}",
                    objectPath, headerName, values);
        } else {
            log.info("[DEBUG applyGetHeadersHint] failed to apply getHeaders hint for path={}, headerName={}",
                    objectPath, headerName);
        }
    }

    private boolean installGetHeadersStub(Object root,
                                      String objectPath,
                                      String headerName,
                                      List<String> values,
                                      ClassLoader loader) throws Exception {
        if (root == null || objectPath == null || objectPath.isEmpty()) {
            return false;
        }

        ParentAndField pf = resolveParentAndField(root, objectPath);
        if (pf == null || pf.field == null) {
            return false;
        }

        Field field = pf.field;
        field.setAccessible(true);

        Class<?> fieldType = field.getType();

        // Prefer a dynamic proxy only when the field type is an interface.
        if (fieldType.isInterface()) {
            Object proxy = Proxy.newProxyInstance(
                    loader,
                    new Class<?>[]{fieldType},
                    (proxyObj, method, args) -> {
                        String methodName = method.getName();

                        // getHeaders("Cookie") -> List<String>
                        if ("getHeaders".equals(methodName)
                                && args != null
                                && args.length == 1
                                && Objects.equals(String.valueOf(args[0]), headerName)) {
                            return new ArrayList<>(values);
                        }

                        // getHeader("Cookie") -> first value.
                        if ("getHeader".equals(methodName)
                                && args != null
                                && args.length == 1
                                && Objects.equals(String.valueOf(args[0]), headerName)) {
                            return values.isEmpty() ? null : values.get(0);
                        }

                        // containsHeader("Cookie") -> true
                        if ("containsHeader".equals(methodName)
                                && args != null
                                && args.length == 1
                                && Objects.equals(String.valueOf(args[0]), headerName)) {
                            return true;
                        }

                        // toString / hashCode / equals
                        if ("toString".equals(methodName) && (args == null || args.length == 0)) {
                            return "HeaderStub(" + headerName + "=" + values + ")";
                        }
                        if ("hashCode".equals(methodName) && (args == null || args.length == 0)) {
                            return System.identityHashCode(proxyObj);
                        }
                        if ("equals".equals(methodName) && args != null && args.length == 1) {
                            return proxyObj == args[0];
                        }

                        // Return default values for the remaining methods.
                        return defaultValueForType(method.getReturnType());
                    }
            );

            field.set(pf.parent, proxy);
            return true;
        }

        // If the field type is not an interface, leave it unchanged here.
        return false;
    }

    private ParentAndField resolveParentAndField(Object root, String path) throws Exception {
        if (root == null || path == null || path.isEmpty()) {
            return null;
        }
    
        String normalized = path;
        if (normalized.startsWith("this.")) {
            normalized = normalized.substring("this.".length());
        } else if (normalized.equals("this")) {
            return null;
        }
    
        String[] parts = normalized.split("\\.");
        if (parts.length == 0) {
            return null;
        }
    
        Object current = root;
        for (int i = 0; i < parts.length - 1; i++) {
            Field f = tryFindField(current.getClass(), parts[i]);
            if (f == null) {
                return null;
            }
            f.setAccessible(true);
            current = f.get(current);
            if (current == null) {
                return null;
            }
        }
    
        Field leafField = tryFindField(current.getClass(), parts[parts.length - 1]);
        if (leafField == null) {
            return null;
        }
    
        ParentAndField result = new ParentAndField();
        result.parent = current;
        result.field = leafField;
        return result;
    }

    private Object defaultValueForType(Class<?> returnType) {
        if (returnType == null || returnType == Void.TYPE) {
            return null;
        }
        if (!returnType.isPrimitive()) {
            return null;
        }
        if (returnType == boolean.class) return false;
        if (returnType == byte.class) return (byte) 0;
        if (returnType == short.class) return (short) 0;
        if (returnType == int.class) return 0;
        if (returnType == long.class) return 0L;
        if (returnType == float.class) return 0f;
        if (returnType == double.class) return 0d;
        if (returnType == char.class) return '\0';
        return null;
    }

    private void applyHeadersMethodHint(Object root,
                                    String rawKey,
                                    Object rawValue,
                                    Map<String, Object> fieldValues,
                                    ClassLoader loader) throws Exception {
        // Example:
        // headers(COOKIE_HEADER)
        String inside = rawKey.substring(rawKey.indexOf('(') + 1, rawKey.lastIndexOf(')')).trim();

        String headerName;
        Object aliasValue = fieldValues.get(inside);
        if (aliasValue != null) {
            headerName = String.valueOf(aliasValue);
        } else {
            headerName = stripQuotes(inside);
        }

        List<String> values = toStringList(rawValue);
        if (values.isEmpty()) {
            return;
        }

        boolean applied = tryApplyHeaderValues(root, headerName, values);
        if (!applied) {
            log.info("[DEBUG applyHeadersMethodHint] failed to hydrate root={} for key={}",
                    root.getClass().getName(), rawKey);
        } else {
            log.info("[DEBUG applyHeadersMethodHint] applied headers(...) hint on root={}, headerName={}, values={}",
                    root.getClass().getName(), headerName, values);
        }
    }

    private void applyHeaderValuesAliasHint(Object root,
                                        String rawKey,
                                        Object rawValue,
                                        Map<String, Object> fieldValues,
                                        ClassLoader loader) throws Exception {
        SortedMap<Integer, String> indexed = collectIndexedStringValues("headerValues", fieldValues);
        if (indexed.isEmpty()) {
            return;
        }

        List<String> values = new ArrayList<>(indexed.values());

        String headerName = "Cookie";
        Object cookieHeaderAlias = fieldValues.get("COOKIE_HEADER");
        if (cookieHeaderAlias != null) {
            headerName = String.valueOf(cookieHeaderAlias);
        }

        boolean applied = tryApplyHeaderValues(root, headerName, values);
        if (!applied) {
            log.info("[DEBUG applyHeaderValuesAliasHint] failed to hydrate root={} for headerValues alias",
                    root.getClass().getName());
        } else {
            log.info("[DEBUG applyHeaderValuesAliasHint] applied headerValues alias on root={}, headerName={}, values={}",
                    root.getClass().getName(), headerName, values);
        }
    }
    private boolean tryApplyHeaderValues(Object receiver,
                                     String headerName,
                                     List<String> values) {
        if (receiver == null || headerName == null || values == null || values.isEmpty()) {
            return false;
        }

        // 1) Prefer addHeader(String, Object).
        if (invokeHeaderMutationMethod(receiver, "addHeader", headerName, values)) {
            return true;
        }

        // 2) Then try setHeader(String, Object).
        if (invokeHeaderMutationMethod(receiver, "setHeader", headerName, values)) {
            return true;
        }

        // 3) Then try header(String, String) or withHeader(String, String).
        if (invokeSingleValueHeaderMethod(receiver, "header", headerName, values)) {
            return true;
        }
        if (invokeSingleValueHeaderMethod(receiver, "withHeader", headerName, values)) {
            return true;
        }

        // 4) If the receiver has an httpRequest field, recursively apply the header to the inner object.
        Field nestedHttpRequestField = tryFindField(receiver.getClass(), "httpRequest");
        if (nestedHttpRequestField != null) {
            try {
                nestedHttpRequestField.setAccessible(true);
                Object nested = nestedHttpRequestField.get(receiver);
                if (nested != null && tryApplyHeaderValues(nested, headerName, values)) {
                    return true;
                }
            } catch (Exception ignored) {
            }
        }

        // 5) Finally, try writing the headers field directly as a fallback.
        Field headersField = tryFindField(receiver.getClass(), "headers");
        if (headersField != null) {
            try {
                headersField.setAccessible(true);
                Object current = headersField.get(receiver);

                if (current instanceof Map) {
                    @SuppressWarnings("unchecked")
                    Map<Object, Object> map = (Map<Object, Object>) current;
                    map.put(headerName, new ArrayList<>(values));
                    return true;
                }

                if (current == null && Map.class.isAssignableFrom(headersField.getType())) {
                    Map<String, List<String>> map = new LinkedHashMap<>();
                    map.put(headerName, new ArrayList<>(values));
                    headersField.set(receiver, map);
                    return true;
                }
            } catch (Exception ignored) {
            }
        }

        return false;
    }
    private Object resolveObjectPath(Object root, String path) throws Exception {
        if (root == null || path == null || path.isEmpty()) {
            return null;
        }
    
        String normalized = path;
        if (normalized.startsWith("this.")) {
            normalized = normalized.substring("this.".length());
        } else if (normalized.equals("this")) {
            return root;
        }
    
        if (normalized.isEmpty()) {
            return root;
        }
    
        String[] parts = normalized.split("\\.");
        Object current = root;
        for (String part : parts) {
            if (current == null) {
                return null;
            }
            Field field = tryFindField(current.getClass(), part);
            if (field == null) {
                return null;
            }
            field.setAccessible(true);
            current = field.get(current);
        }
        return current;
    }
    private String stripQuotes(String s) {
        if (s == null) {
            return null;
        }
        String t = s.trim();
        if ((t.startsWith("\"") && t.endsWith("\"")) || (t.startsWith("'") && t.endsWith("'"))) {
            return t.substring(1, t.length() - 1);
        }
        return t;
    }
    private List<String> toStringList(Object rawValue) {
        List<String> result = new ArrayList<>();
        if (rawValue == null) {
            return result;
        }
    
        if (rawValue instanceof List<?>) {
            for (Object o : (List<?>) rawValue) {
                if (o != null) {
                    result.add(String.valueOf(o));
                }
            }
            return result;
        }
    
        result.add(String.valueOf(rawValue));
        return result;
    }
    private SortedMap<Integer, String> collectIndexedStringValues(String aliasPrefix,
                                                              Map<String, Object> fieldValues) {
        SortedMap<Integer, String> indexed = new TreeMap<>();
        String prefix = aliasPrefix + "[";

        for (Map.Entry<String, Object> entry : fieldValues.entrySet()) {
            String key = entry.getKey();
            if (!key.startsWith(prefix) || !key.endsWith("]")) {
                continue;
            }

            String inside = key.substring(prefix.length(), key.length() - 1).trim();
            if (!inside.matches("\\d+")) {
                continue;
            }

            int index = Integer.parseInt(inside);
            indexed.put(index, entry.getValue() == null ? null : String.valueOf(entry.getValue()));
        }

        return indexed;
    }
    private boolean invokeHeaderMutationMethod(Object receiver,
                                           String methodName,
                                           String headerName,
                                           List<String> values) {
        for (Method method : receiver.getClass().getMethods()) {
            if (!method.getName().equals(methodName)) {
                continue;
            }
            if (method.getParameterCount() != 2) {
                continue;
            }

            Class<?>[] pts = method.getParameterTypes();
            if (pts[0] != String.class) {
                continue;
            }

            try {
                method.setAccessible(true);

                // The second parameter is a Collection/List.
                if (Collection.class.isAssignableFrom(pts[1])) {
                    method.invoke(receiver, headerName, new ArrayList<>(values));
                    return true;
                }

                // The second parameter is an array.
                if (pts[1].isArray() && pts[1].getComponentType() == String.class) {
                    method.invoke(receiver, headerName, values.toArray(new String[0]));
                    return true;
                }

                // The second parameter is a single value, so invoke once per value.
                for (String v : values) {
                    method.invoke(receiver, headerName, v);
                }
                return true;
            } catch (Exception ignored) {
            }
        }
        return false;
    }
    private boolean invokeSingleValueHeaderMethod(Object receiver,
                                              String methodName,
                                              String headerName,
                                              List<String> values) {
        for (Method method : receiver.getClass().getMethods()) {
            if (!method.getName().equals(methodName)) {
                continue;
            }
            if (method.getParameterCount() != 2) {
                continue;
            }

            Class<?>[] pts = method.getParameterTypes();
            if (pts[0] != String.class || pts[1] != String.class) {
                continue;
            }

            try {
                method.setAccessible(true);
                for (String v : values) {
                    method.invoke(receiver, headerName, v);
                }
                return true;
            } catch (Exception ignored) {
            }
        }
        return false;
    }

}
