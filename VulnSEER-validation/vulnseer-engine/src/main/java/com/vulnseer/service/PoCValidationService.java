package com.vulnseer.service;

import cn.hutool.core.date.BetweenFormatter;
import cn.hutool.core.date.DateUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.FileWriter;
import cn.hutool.core.lang.Pair;
import cn.hutool.http.HtmlUtil;
import com.alibaba.fastjson2.JSON;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vulnseer.config.ConfigProperty;
import com.vulnseer.config.GlobalConfiguration;
import com.vulnseer.config.ProjectContext;
import com.vulnseer.dependency.MavenDependency;
import com.vulnseer.dependency.MavenDependencyTree;
import com.vulnseer.fuzz.FuzzClassLoader;
import com.vulnseer.fuzz.result.InvokeMethodResult;
import com.vulnseer.report.html.HTMLFuzzChainResult;
import com.vulnseer.report.html.HTMLFuzzChainResultWrapper;
import com.vulnseer.report.html.HTMLFuzzStepResult;
import com.vulnseer.report.html.HTMLPage;
import com.vulnseer.report.json.JSONFuzzChainResult;
import com.vulnseer.report.json.JSONFuzzChainResultWrapper;
import com.vulnseer.report.json.JSONFuzzStepResult;
import com.vulnseer.report.json.JSONReport;
import com.vulnseer.testcase.TestCaseService;
import com.vulnseer.testcase.model.MetaInfo;
import com.vulnseer.testcase.model.TestcaseUnit;
import com.vulnseer.testcase.runner.TestcaseRunner;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.velocity.Template;
import org.apache.velocity.VelocityContext;
import org.apache.velocity.app.Velocity;

import java.io.File;
import java.io.IOException;
import java.io.StringWriter;
import java.lang.reflect.Array;
import java.util.*;

@Slf4j
public class PoCValidationService {

    private final ProjectContext projectContext;
    private final ConfigProperty configProperty;
    private final ObjectMapper objectMapper;
    private final TestCaseService testCaseService;
    private final OracleValidationService oracleValidationService;
    private final Map<String, ClassLoader> testcaseLoaderCache;

    public PoCValidationService(ProjectContext projectContext, ConfigProperty configProperty) {
        this.projectContext = projectContext;
        this.configProperty = normalizeConfigProperty(configProperty);
        this.objectMapper = new ObjectMapper();
        this.testCaseService = new TestCaseService();
        this.testCaseService.setConfigProperty(this.configProperty);
        this.oracleValidationService = new OracleValidationService();
        this.testcaseLoaderCache = new HashMap<>();
    }

    /**
     * Backward-compatible API:
     * - If input.json is a single object, return one report.
     * - If input.json is an array, return the first report.
     */
    public ValidationReport validate(File inputJsonFile) {
        List<ValidationReport> reports = validateAll(inputJsonFile);
        if (reports.isEmpty()) {
            throw new RuntimeException("No validation case found in input json: " + inputJsonFile.getAbsolutePath());
        }
        return reports.get(0);
    }

    /**
     * New API: support multiple call-chain cases in one input.json file.
     */
    public List<ValidationReport> validateAll(File inputJsonFile) {
        List<UnifiedInput> inputs = loadValidationInputs(inputJsonFile);
        List<ValidationReport> reports = new ArrayList<>();

        for (UnifiedInput input : inputs) {
            PoCReplayRunner.ValidationInput replayInput = new PoCReplayRunner.ValidationInput();
            replayInput.setEntryMethod(input.getEntryMethod());
            replayInput.setSinkMethod(input.getSinkMethod());

            PoCReplayRunner.FinalPayload finalPayload = new PoCReplayRunner.FinalPayload();
            finalPayload.setArgValues(input.getFinalPayload().getArgValues());
            finalPayload.setFieldValues(input.getFinalPayload().getFieldValues());
            replayInput.setFinalPayload(finalPayload);

            try {
                PoCReplayRunner replayRunner = new PoCReplayRunner(projectContext);
                PoCReplayRunner.ReplayResult replayResult = replayRunner.replay(replayInput);
                OracleValidationResult oracleResult = validateOracle(input, replayResult);
                reports.add(buildValidationReport(input, replayResult, oracleResult));
            } catch (Throwable e) {
                log.error("[DEBUG validateAll] replay failed for {} {}",
                        input.getCveName(),
                        input.getChainId(),
                        e);
                reports.add(buildFrameworkErrorReport(input, e));
            }
        }

        return reports;
    }

    /**
     * Backward-compatible API for generating a single report.
     */
    public void generateValidationReport(ValidationReport report) {
        generateValidationReport(Collections.singletonList(report));
    }

    /**
     * New API: generate one report.json and one report.html for multiple reports.
     */
    public void generateValidationReport(List<ValidationReport> reports) {
        generateJsonReport(reports);
        generateHtmlReport(reports);
    }

    private ValidationReport buildFrameworkErrorReport(UnifiedInput input, Throwable e) {
        ValidationReport report = new ValidationReport();
        report.setCveName(input.getCveName());
        report.setChainId(input.getChainId());
        report.setCallChain(input.getCallChain());
        report.setEntryMethod(input.getEntryMethod());
        report.setSinkMethod(input.getSinkMethod());
        report.setResultStatus("FrameworkError");
        report.setSinkReached(false);
        report.setVulnerabilityTriggered(false);
        report.setSuccessStep(0);
        report.setFuzzChainTime(0L);
        report.setArgsMapList(Collections.emptyList());
        report.setFieldMap(Collections.singletonMap("value", "null"));

        Map<String, String> invokeMap = new LinkedHashMap<>();
        invokeMap.put("Return Value", "null");
        invokeMap.put("Throw Exception", HtmlUtil.escape(String.valueOf(e)));
        invokeMap.put("Reached Result", "NotReached");
        invokeMap.put("Verified Result", OracleValidationResult.Status.ORACLE_ERROR.name());
        invokeMap.put("Oracle Message", HtmlUtil.escape(String.valueOf(e)));
        report.setInvokeResultMap(invokeMap);

        report.setNumDecisions(input.getNumDecisions());
        report.setNumInclude(input.getNumInclude());
        report.setNumExclude(input.getNumExclude());
        report.setTruncated(input.getTruncated());
        report.setElapsedSec(input.getElapsedSec());
        report.setDescription(input.getFinalPayload() == null ? null : input.getFinalPayload().getDescription());
        report.setRawInput(input.getRawInput());
        report.setNotes("Framework-level validation error.");
        return report;
    }

    private void generateJsonReport(List<ValidationReport> reports) {
        JSONReport jsonReport = buildJsonReport(reports);
        String jsonString = JSON.toJSONString(jsonReport);
        String outputPath = GlobalConfiguration.OUTPUT_DIR_PATH + File.separator + "report.json";
        FileWriter writer = new FileWriter(outputPath);
        writer.write(jsonString);
        log.info("validation json report written to {}", outputPath);
    }

    private void generateHtmlReport(List<ValidationReport> reports) {
        try {
            Properties properties = new Properties();
            properties.put("resource.loader.file.class", "org.apache.velocity.runtime.resource.loader.ClasspathResourceLoader");
            Velocity.init(properties);

            VelocityContext velocityContext = new VelocityContext();
            velocityContext.put("title", "Fuzzing Report");

            HTMLPage htmlPage = buildHtmlPage(reports);
            velocityContext.put("context", htmlPage);

            Template template = Velocity.getTemplate("report.vm", "utf-8");

            StringWriter sw = new StringWriter();
            template.merge(velocityContext, sw);

            String outputPath = GlobalConfiguration.OUTPUT_DIR_PATH + File.separator + "report.html";
            FileWriter writer = new FileWriter(outputPath);
            writer.write(sw.toString());

            log.info("validation html report written to {}", outputPath);
        } catch (Exception e) {
            throw new RuntimeException("Generate validation html report failed", e);
        }
    }

    private JSONReport buildJsonReport(List<ValidationReport> reports) {
        JSONReport jsonReport = new JSONReport();
        jsonReport.setProjectPath(projectContext.getProjectPath());

        int chainNum = reports == null ? 0 : reports.size();
        jsonReport.setVulDependencyChainNum(chainNum);
        jsonReport.setVulMethodCallChainNum(chainNum);

        double avgChainLen = 0D;
        int totalSuccessStep = 0;
        int reachedCount = 0;
        int triggeredCount = 0;
        List<JSONFuzzChainResultWrapper> wrapperList = new ArrayList<>();

        if (reports != null && !reports.isEmpty()) {
            int totalLen = 0;
            for (ValidationReport report : reports) {
                totalLen += report.getCallChain() == null ? 0 : report.getCallChain().size();
                totalSuccessStep += report.getSuccessStep() == null ? 0 : report.getSuccessStep();
                if (report.isSinkReached()) {
                    reachedCount++;
                }
                if (report.isVulnerabilityTriggered()) {
                    triggeredCount++;
                }

                JSONFuzzStepResult step = new JSONFuzzStepResult();
                step.setConsumeTime(report.getFuzzChainTime() == null ? 0L : report.getFuzzChainTime());
                step.setConsumeTimeStr(DateUtil.formatBetween(step.getConsumeTime(), BetweenFormatter.Level.SECOND));
                step.setArgsMapList(report.getArgsMapList());
                step.setFieldMap(report.getFieldMap());
                step.setInvokeResultMap(report.getInvokeResultMap());

                JSONFuzzChainResult fuzzChainResult = new JSONFuzzChainResult();
                fuzzChainResult.setSuccessStep(report.getSuccessStep() == null ? 0 : report.getSuccessStep());
                fuzzChainResult.setFuzzChainTime(report.getFuzzChainTime() == null ? 0L : report.getFuzzChainTime());
                fuzzChainResult.setFuzzChainTimeStr(DateUtil.formatBetween(
                        fuzzChainResult.getFuzzChainTime(),
                        BetweenFormatter.Level.SECOND
                ));
                fuzzChainResult.setFuzzStepList(Collections.singletonList(step));

                JSONFuzzChainResultWrapper wrapper = new JSONFuzzChainResultWrapper();
                wrapper.setMethodCallList(report.getCallChain());

                Map<String, JSONFuzzChainResult> fuzzResultMap = new LinkedHashMap<>();
                String cveKey = buildResultKey(report);
                fuzzResultMap.put(cveKey, fuzzChainResult);
                wrapper.setFuzzResultMap(fuzzResultMap);

                wrapperList.add(wrapper);
            }
            avgChainLen = totalLen * 1.0 / reports.size();
        }

        jsonReport.setAverageMethodCallChainLength(avgChainLen);
        jsonReport.setReachedCount(reachedCount);
        jsonReport.setTriggeredCount(triggeredCount);

        jsonReport.setAnalysisTime(projectContext.getAnalysisTime());
        jsonReport.setAnalysisTimeStr(DateUtil.formatBetween(projectContext.getAnalysisTime(), BetweenFormatter.Level.SECOND));

        jsonReport.setFuzzTime(projectContext.getFuzzTime());
        jsonReport.setFuzzTimeStr(DateUtil.formatBetween(projectContext.getFuzzTime(), BetweenFormatter.Level.SECOND));

        jsonReport.setTotalTime(projectContext.getTotalTime());
        jsonReport.setTotalTimeStr(DateUtil.formatBetween(projectContext.getTotalTime(), BetweenFormatter.Level.SECOND));

        jsonReport.setTotalSuccessStep(totalSuccessStep);
        jsonReport.setAverageSuccessStep(chainNum == 0 ? 0D : totalSuccessStep * 1.0 / chainNum);

        jsonReport.setFuzzChainList(wrapperList);
        return jsonReport;
    }

    private HTMLPage buildHtmlPage(List<ValidationReport> reports) {
        HTMLPage page = new HTMLPage();
        page.setProjectPath(projectContext.getProjectPath());
        page.setVulDependencyList(buildVulDependencyList(reports));
        page.setTotalTime(DateUtil.formatBetween(projectContext.getTotalTime(), BetweenFormatter.Level.SECOND));
        page.setReachedCount(countReached(reports));
        page.setTriggeredCount(countTriggered(reports));

        List<HTMLFuzzChainResultWrapper> wrapperList = new ArrayList<>();
        if (reports != null) {
            for (ValidationReport report : reports) {
                HTMLFuzzStepResult step = new HTMLFuzzStepResult();
                step.setConsumeTime(DateUtil.formatBetween(
                        report.getFuzzChainTime() == null ? 0L : report.getFuzzChainTime(),
                        BetweenFormatter.Level.SECOND
                ));
                step.setArgsMapList(report.getArgsMapList());
                step.setFieldMap(report.getFieldMap());
                step.setInvokeResultMap(report.getInvokeResultMap());

                HTMLFuzzChainResult chainResult = new HTMLFuzzChainResult();
                chainResult.setSuccessStep(report.getSuccessStep() == null ? 0 : report.getSuccessStep());
                chainResult.setFuzzStepList(Collections.singletonList(step));

                HTMLFuzzChainResultWrapper wrapper = new HTMLFuzzChainResultWrapper();
                wrapper.setMethodCallList(escapeMethodCallList(report.getCallChain()));

                Map<String, HTMLFuzzChainResult> fuzzResultMap = new LinkedHashMap<>();
                String cveKey = buildResultKey(report);
                fuzzResultMap.put(cveKey, chainResult);
                wrapper.setFuzzResultMap(fuzzResultMap);

                wrapperList.add(wrapper);
            }
        }

        page.setFuzzChainList(wrapperList);
        return page;
    }

    private List<String> buildVulDependencyList(List<ValidationReport> reports) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        if (reports != null) {
            for (ValidationReport report : reports) {
                if (report.getCveName() != null && !report.getCveName().isEmpty()) {
                    set.add(report.getCveName());
                }
            }
        }
        return new ArrayList<>(set);
    }

    private List<String> escapeMethodCallList(List<String> callChain) {
        List<String> escaped = new ArrayList<>();
        if (callChain != null) {
            for (String methodCall : callChain) {
                escaped.add(methodCall == null ? "" : HtmlUtil.escape(methodCall));
            }
        }
        return escaped;
    }

    private String buildResultKey(ValidationReport report) {
        String cve = report.getCveName() == null ? "UNKNOWN-CVE" : report.getCveName();
        String chainId = report.getChainId();
        if (chainId == null || chainId.isEmpty()) {
            return cve;
        }
        return cve + " [" + chainId + "]";
    }

    private ValidationReport buildValidationReport(UnifiedInput input,
                                                   PoCReplayRunner.ReplayResult replayResult,
                                                   OracleValidationResult oracleResult) {
        ValidationReport report = new ValidationReport();
        report.setCveName(input.getCveName());
        report.setChainId(input.getChainId());
        report.setCallChain(input.getCallChain());
        report.setEntryMethod(input.getEntryMethod());
        report.setSinkMethod(input.getSinkMethod());

        report.setResultStatus(replayResult.getResultStatus());
        report.setSinkReached(replayResult.isSinkReached());
        report.setVulnerabilityTriggered(oracleResult != null && oracleResult.isTriggered());

        report.setNumDecisions(input.getNumDecisions());
        report.setNumInclude(input.getNumInclude());
        report.setNumExclude(input.getNumExclude());
        report.setTruncated(input.getTruncated());
        report.setElapsedSec(input.getElapsedSec());
        report.setDescription(input.getFinalPayload() == null ? null : input.getFinalPayload().getDescription());
        report.setRawInput(input.getRawInput());

        report.setSuccessStep(replayResult.isSinkReached() ? Math.max(0, input.getCallChain().size() - 1) : 0);
        report.setFuzzChainTime(Math.round((input.getElapsedSec() == null ? 0D : input.getElapsedSec()) * 1000));

        report.setArgsMapList(buildArgsMapList(replayResult.getEntryArgs()));
        report.setFieldMap(getFieldMap(replayResult.getReplayTarget()));
        report.setInvokeResultMap(buildInvokeResultMap(
                replayResult.getInvokeMethodResult(), replayResult.isSinkReached(), oracleResult));

        report.setNotes(buildNotes(replayResult, oracleResult));

        return report;
    }

    private OracleValidationResult validateOracle(UnifiedInput input, PoCReplayRunner.ReplayResult replayResult) {
        if (replayResult == null || !replayResult.isSinkReached()) {
            return OracleValidationResult.notReached();
        }

        List<ResolvedOracle> resolvedOracles;
        try {
            resolvedOracles = findMatchingOracles(input);
        } catch (Throwable throwable) {
            return OracleValidationResult.error(null, throwable);
        }

        if (resolvedOracles.isEmpty()) {
            return OracleValidationResult.missing("No groundtruth oracle matches cve_name and sink_method.");
        }

        OracleValidationResult bestResult = null;
        for (ResolvedOracle resolvedOracle : resolvedOracles) {
            TestcaseUnit testcaseUnit = resolvedOracle.getTestcaseUnit();
            try {
                prepareTestcaseLoader(resolvedOracle);
                OracleValidationResult oracleResult = oracleValidationService.validate(
                        projectContext.getInstrumentedFuzzClassLoader(),
                        testcaseUnit,
                        replayResult.getInvokeMethodResult(),
                        null,
                        false
                );
                if (oracleResult.isTriggered()) {
                    return oracleResult;
                }
                bestResult = chooseBetterOracleResult(bestResult, oracleResult);
            } catch (Throwable throwable) {
                bestResult = chooseBetterOracleResult(
                        bestResult, OracleValidationResult.error(testcaseUnit, throwable));
            }
        }

        return bestResult == null
                ? OracleValidationResult.missing("No groundtruth oracle could be evaluated.")
                : bestResult;
    }

    private List<ResolvedOracle> findMatchingOracles(UnifiedInput input) {
        List<ResolvedOracle> matches = new ArrayList<>();
        String cveName = input.getCveName();
        String sinkMethod = input.getSinkMethod();

        for (MetaInfo metaInfo : testCaseService.getAllMetaInfo()) {
            if (cveName != null && !cveName.isEmpty() && !Objects.equals(cveName, metaInfo.getVulName())) {
                continue;
            }

            List<TestcaseUnit> testcaseUnits = metaInfo.getTestcaseUnitList();
            if (testcaseUnits == null) {
                continue;
            }

            for (TestcaseUnit testcaseUnit : testcaseUnits) {
                if (isSameMethodSignature(sinkMethod, testcaseUnit.getVulMethodSignature())) {
                    matches.add(new ResolvedOracle(metaInfo, testcaseUnit));
                }
            }
        }
        return matches;
    }

    private void prepareTestcaseLoader(ResolvedOracle resolvedOracle) throws Exception {
        ClassLoader classLoader = projectContext.getInstrumentedFuzzClassLoader();
        if (!(classLoader instanceof FuzzClassLoader)) {
            throw new RuntimeException("Instrumented classloader is not FuzzClassLoader.");
        }

        TestcaseUnit testcaseUnit = resolvedOracle.getTestcaseUnit();
        File storageDir = testcaseUnit.getStorageDir();
        if (storageDir == null || !storageDir.isDirectory()) {
            throw new RuntimeException("Invalid testcase storage directory for " + testcaseUnit.getTestcaseClassName());
        }

        String version = resolveDependencyVersion(resolvedOracle);
        String cacheKey = storageDir.getAbsolutePath() + "#" + version;
        ClassLoader testcaseLoader = testcaseLoaderCache.get(cacheKey);
        if (testcaseLoader == null) {
            testcaseLoader = new TestcaseRunner(storageDir).prepareTestcaseLoader(version);
            testcaseLoaderCache.put(cacheKey, testcaseLoader);
        }

        ((FuzzClassLoader) classLoader).setTestcaseLoader(testcaseLoader);
    }

    private String resolveDependencyVersion(ResolvedOracle resolvedOracle) {
        TestcaseUnit testcaseUnit = resolvedOracle.getTestcaseUnit();
        MavenDependencyTree dependencyTree = projectContext.getDependencyTree();
        if (dependencyTree != null) {
            String version = dependencyTree.getVersionMap().get(
                    new Pair<>(testcaseUnit.getGroupId(), testcaseUnit.getArtifactId()));
            if (version != null) {
                return version;
            }

            MavenDependency rootNode = dependencyTree.getRootNode();
            if (rootNode != null
                    && Objects.equals(rootNode.getGroupId(), testcaseUnit.getGroupId())
                    && Objects.equals(rootNode.getArtifactId(), testcaseUnit.getArtifactId())) {
                return rootNode.getVersion();
            }
        }

        MetaInfo metaInfo = resolvedOracle.getMetaInfo();
        if (metaInfo.getAffectedVersion() != null && !metaInfo.getAffectedVersion().isEmpty()) {
            String fallbackVersion = metaInfo.getAffectedVersion().get(0);
            log.warn("Cannot resolve dependency version for {}:{}, fallback to first affected version {}",
                    testcaseUnit.getGroupId(), testcaseUnit.getArtifactId(), fallbackVersion);
            return fallbackVersion;
        }

        throw new RuntimeException("Cannot resolve dependency version for "
                + testcaseUnit.getGroupId() + ":" + testcaseUnit.getArtifactId());
    }

    private OracleValidationResult chooseBetterOracleResult(OracleValidationResult current,
                                                            OracleValidationResult candidate) {
        if (current == null) {
            return candidate;
        }
        if (candidate == null) {
            return current;
        }
        if (candidate.isTriggered()) {
            return candidate;
        }
        if (OracleValidationResult.Status.ORACLE_NOT_TRIGGERED.equals(candidate.getStatus())
                && !OracleValidationResult.Status.ORACLE_NOT_TRIGGERED.equals(current.getStatus())) {
            return candidate;
        }
        if (OracleValidationResult.Status.INPUT_MISMATCH.equals(candidate.getStatus())
                && OracleValidationResult.Status.ORACLE_ERROR.equals(current.getStatus())) {
            return candidate;
        }
        return current;
    }

    private String buildNotes(PoCReplayRunner.ReplayResult replayResult, OracleValidationResult oracleResult) {
        if (replayResult == null || !replayResult.isSinkReached()) {
            return "Sink method not reached according to instrumented state table.";
        }
        if (oracleResult == null) {
            return "Sink method reached, but oracle validation did not run.";
        }
        return "Sink method reached; oracle status = " + oracleResult.getStatus().name() + ".";
    }

    private String renderOracleStatus(OracleValidationResult oracleResult) {
        if (oracleResult == null) {
            return null;
        }
        return oracleResult.getStatus().name();
    }

    private String renderOracleTestcase(OracleValidationResult oracleResult) {
        if (oracleResult == null || oracleResult.getTestcaseUnit() == null) {
            return null;
        }
        TestcaseUnit testcaseUnit = oracleResult.getTestcaseUnit();
        return testcaseUnit.getTestcaseClassName() + "#" + testcaseUnit.getTestcaseMethodName();
    }

    private int countReached(List<ValidationReport> reports) {
        if (reports == null) {
            return 0;
        }
        int count = 0;
        for (ValidationReport report : reports) {
            if (report.isSinkReached()) {
                count++;
            }
        }
        return count;
    }

    private int countTriggered(List<ValidationReport> reports) {
        if (reports == null) {
            return 0;
        }
        int count = 0;
        for (ValidationReport report : reports) {
            if (report.isVulnerabilityTriggered()) {
                count++;
            }
        }
        return count;
    }

    private List<Map<String, String>> buildArgsMapList(Object[] args) {
        List<Map<String, String>> argsMapList = new ArrayList<>();
        if (args == null) {
            return argsMapList;
        }
        for (Object arg : args) {
            argsMapList.add(getFieldMap(arg));
        }
        return argsMapList;
    }

    private Map<String, String> buildInvokeResultMap(InvokeMethodResult invokeResult,
                                                     boolean sinkReached,
                                                     OracleValidationResult oracleResult) {
        return getInvokeResultMap(invokeResult == null ? new InvokeMethodResult() : invokeResult,
                sinkReached, oracleResult);
    }

    private Map<String, String> getInvokeResultMap(InvokeMethodResult invokeResult,
                                                   boolean sinkReached,
                                                   OracleValidationResult oracleResult) {
        Map<String, String> invokeResultMap = new LinkedHashMap<>();

        Object returnValue = invokeResult.getReturnValue();
        Throwable throwValue = invokeResult.getThrowValue();

        if (returnValue != null) {
            invokeResultMap.put("Return Value", HtmlUtil.escape(String.valueOf(returnValue)));
        } else {
            invokeResultMap.put("Return Value", "null");
        }

        if (throwValue != null) {
            invokeResultMap.put("Throw Exception", HtmlUtil.escape(String.valueOf(throwValue)));
        } else {
            invokeResultMap.put("Throw Exception", "null");
        }

        invokeResultMap.put("Reached Result", sinkReached ? "Reached" : "NotReached");
        invokeResultMap.put("Verified Result", renderOracleStatus(oracleResult));
        String oracleTestcase = renderOracleTestcase(oracleResult);
        if (oracleTestcase != null) {
            invokeResultMap.put("Oracle Testcase", HtmlUtil.escape(oracleTestcase));
        }
        String oracleMessage = oracleResult == null ? null : oracleResult.getMessage();
        if (oracleMessage != null) {
            invokeResultMap.put("Oracle Message", HtmlUtil.escape(oracleMessage));
        }
        return invokeResultMap;
    }

    private Map<String, String> getFieldMap(Object object) {
        Map<String, String> fieldMap = new LinkedHashMap<>();

        if (object == null) {
            fieldMap.put("value", "null");
            return fieldMap;
        }

        if (isSimpleDisplayType(object)) {
            fieldMap.put("value", safeDisplay(object));
            return fieldMap;
        }

        Class<?> clazz = object.getClass();

        if (clazz.isArray()) {
            fieldMap.put("value", renderArray(object));
            return fieldMap;
        }

        if (object instanceof Collection<?>) {
            fieldMap.put("value", safeDisplay(object));
            return fieldMap;
        }

        if (object instanceof Map<?, ?>) {
            fieldMap.put("value", safeDisplay(object));
            return fieldMap;
        }

        try {
            java.lang.reflect.Field[] fields = clazz.getDeclaredFields();
            boolean hasField = false;

            for (java.lang.reflect.Field field : fields) {
                if (java.lang.reflect.Modifier.isFinal(field.getModifiers())
                        && java.lang.reflect.Modifier.isStatic(field.getModifiers())) {
                    continue;
                }

                field.setAccessible(true);
                Object value = field.get(object);

                if (value == null) {
                    fieldMap.put(field.getName(), "null");
                } else if (isSimpleDisplayType(value)) {
                    fieldMap.put(field.getName(), safeDisplay(value));
                } else if (value.getClass().isArray()) {
                    fieldMap.put(field.getName(), renderArray(value));
                } else {
                    fieldMap.put(field.getName(), safeDisplay(value));
                }
                hasField = true;
            }

            if (!hasField) {
                fieldMap.put("value", safeDisplay(object));
            }
        } catch (Throwable e) {
            fieldMap.put("value", safeDisplay(object));
        }
        return fieldMap;
    }

    private boolean isSimpleDisplayType(Object object) {
        if (object == null) {
            return true;
        }
        return object instanceof String
                || object instanceof Number
                || object instanceof Boolean
                || object instanceof Character
                || object.getClass().isEnum()
                || object instanceof Class<?>;
    }

    private String safeDisplay(Object object) {
        return HtmlUtil.escape(String.valueOf(object));
    }

    private String renderArray(Object array) {
        if (array == null) {
            return "null";
        }

        Class<?> componentType = array.getClass().getComponentType();
        if (componentType == byte.class) {
            return HtmlUtil.escape(Arrays.toString((byte[]) array));
        }
        if (componentType == short.class) {
            return HtmlUtil.escape(Arrays.toString((short[]) array));
        }
        if (componentType == int.class) {
            return HtmlUtil.escape(Arrays.toString((int[]) array));
        }
        if (componentType == long.class) {
            return HtmlUtil.escape(Arrays.toString((long[]) array));
        }
        if (componentType == float.class) {
            return HtmlUtil.escape(Arrays.toString((float[]) array));
        }
        if (componentType == double.class) {
            return HtmlUtil.escape(Arrays.toString((double[]) array));
        }
        if (componentType == char.class) {
            return HtmlUtil.escape(Arrays.toString((char[]) array));
        }
        if (componentType == boolean.class) {
            return HtmlUtil.escape(Arrays.toString((boolean[]) array));
        }

        int len = Array.getLength(array);
        List<String> values = new ArrayList<>(len);
        for (int i = 0; i < len; i++) {
            Object item = Array.get(array, i);
            values.add(String.valueOf(item));
        }
        return HtmlUtil.escape(values.toString());
    }

    private ConfigProperty normalizeConfigProperty(ConfigProperty configProperty) {
        ConfigProperty effectiveConfig = configProperty == null ? new ConfigProperty() : configProperty;
        File configuredGroundTruth = FileUtil.file(effectiveConfig.getGroundTruthPath());
        if (configuredGroundTruth.exists() && configuredGroundTruth.isDirectory()) {
            return effectiveConfig;
        }

        File localGroundTruth = FileUtil.file(System.getProperty("user.dir"), "groundtruth");
        if (localGroundTruth.exists() && localGroundTruth.isDirectory()) {
            log.warn("Configured groundtruth path {} does not exist, fallback to {}",
                    configuredGroundTruth.getAbsolutePath(), localGroundTruth.getAbsolutePath());
            effectiveConfig.setGroundTruthPath(localGroundTruth.getAbsolutePath());
        }
        return effectiveConfig;
    }

    private boolean isSameMethodSignature(String left, String right) {
        String normalizedLeft = normalizeMethodSignature(left);
        String normalizedRight = normalizeMethodSignature(right);
        return normalizedLeft != null && normalizedLeft.equals(normalizedRight);
    }

    private String normalizeMethodSignature(String signature) {
        if (signature == null) {
            return null;
        }

        String trimmed = signature.trim();
        if (trimmed.isEmpty()) {
            return null;
        }

        try {
            if (trimmed.startsWith("<") && trimmed.endsWith(">")) {
                return normalizeSootSignature(trimmed);
            }
            return normalizeReplaySignature(trimmed);
        } catch (RuntimeException e) {
            log.warn("Unsupported method signature format: {}", signature);
            return null;
        }
    }

    private String normalizeSootSignature(String signature) {
        String body = signature.substring(1, signature.length() - 1).trim();
        int colon = body.indexOf(':');
        int leftParen = body.indexOf('(');
        int rightParen = body.lastIndexOf(')');
        if (colon < 0 || leftParen < 0 || rightParen < leftParen) {
            return null;
        }

        String className = body.substring(0, colon).trim();
        String returnAndMethod = body.substring(colon + 1, leftParen).trim();
        int methodSep = returnAndMethod.lastIndexOf(' ');
        if (methodSep < 0) {
            return null;
        }

        String returnType = returnAndMethod.substring(0, methodSep).trim();
        String methodName = returnAndMethod.substring(methodSep + 1).trim();
        String params = normalizeParamList(body.substring(leftParen + 1, rightParen));
        return className + "#" + returnType + "#" + methodName + "#" + params;
    }

    private String normalizeReplaySignature(String signature) {
        int colon = signature.indexOf(':');
        int leftParen = signature.indexOf('(');
        int rightParen = signature.lastIndexOf(')');
        if (colon < 0 || leftParen < 0 || rightParen < leftParen) {
            return null;
        }

        String left = signature.substring(0, colon).trim();
        int lastDot = left.lastIndexOf('.');
        if (lastDot < 0) {
            return null;
        }

        String className = left.substring(0, lastDot).trim();
        String methodName = left.substring(lastDot + 1).trim();
        String returnType = signature.substring(colon + 1, leftParen).trim();
        String params = normalizeParamList(signature.substring(leftParen + 1, rightParen));
        return className + "#" + returnType + "#" + methodName + "#" + params;
    }

    private String normalizeParamList(String paramsRaw) {
        if (paramsRaw == null || paramsRaw.trim().isEmpty()) {
            return "";
        }

        List<String> params = new ArrayList<>();
        for (String param : paramsRaw.split(",")) {
            String trimmed = param.trim();
            if (!trimmed.isEmpty()) {
                params.add(trimmed);
            }
        }
        return String.join(",", params);
    }

    /**
     * Supports two input forms:
     * 1) A single JSON object.
     * 2) An array of JSON objects.
     */
    private List<UnifiedInput> loadValidationInputs(File inputJsonFile) {
        try {
            Object root = objectMapper.readValue(inputJsonFile, Object.class);
            List<UnifiedInput> inputs = new ArrayList<>();

            if (root instanceof List<?>) {
                for (Object item : (List<?>) root) {
                    if (!(item instanceof Map)) {
                        throw new RuntimeException("Each element in input json array must be an object");
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, Object> one = (Map<String, Object>) item;
                    inputs.add(parseUnifiedInput(one));
                }
                return inputs;
            }

            if (root instanceof Map<?, ?>) {
                @SuppressWarnings("unchecked")
                Map<String, Object> one = (Map<String, Object>) root;
                inputs.add(parseUnifiedInput(one));
                return inputs;
            }

            throw new RuntimeException("Unsupported input json root type: " + root.getClass().getName());
        } catch (IOException e) {
            throw new RuntimeException("Load unified input json failed: " + inputJsonFile.getAbsolutePath(), e);
        }
    }

    private UnifiedInput parseUnifiedInput(Map<String, Object> root) {
        UnifiedInput input = new UnifiedInput();
        input.setRawInput(root);

        input.setCveName(asString(root.get("cve_name")));
        input.setChainId(asString(root.get("chain_id")));
        input.setEntryMethod(asString(root.get("entry_method")));
        input.setSinkMethod(asString(root.get("sink_method")));
        input.setNumDecisions(asInteger(root.get("num_decisions")));
        input.setNumInclude(asInteger(root.get("num_include")));
        input.setNumExclude(asInteger(root.get("num_exclude")));
        input.setTruncated(asBoolean(root.get("truncated")));
        input.setElapsedSec(asDouble(root.get("elapsed_sec")));

        Object callChainObj = root.get("call_chain");
        if (callChainObj instanceof List) {
            @SuppressWarnings("unchecked")
            List<String> callChain = (List<String>) callChainObj;
            input.setCallChain(callChain);
        } else {
            throw new RuntimeException("input json missing valid 'call_chain'");
        }

        Object finalPayloadObj = root.get("final_payload");
        if (!(finalPayloadObj instanceof Map)) {
            throw new RuntimeException("input json missing valid 'final_payload'");
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> finalPayload = (Map<String, Object>) finalPayloadObj;

        FinalPayload payload = new FinalPayload();
        payload.setDescription(asString(finalPayload.get("description")));

        Object argValuesObj = finalPayload.get("arg_values");
        if (argValuesObj instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> argValues = (Map<String, Object>) argValuesObj;
            payload.setArgValues(argValues);
        } else {
            payload.setArgValues(Collections.emptyMap());
        }

        Object fieldValuesObj = finalPayload.get("field_values");
        if (fieldValuesObj instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> fieldValues = (Map<String, Object>) fieldValuesObj;
            payload.setFieldValues(fieldValues);
        } else {
            payload.setFieldValues(Collections.emptyMap());
        }

        input.setFinalPayload(payload);

        if (input.getEntryMethod() == null || input.getEntryMethod().isEmpty()) {
            input.setEntryMethod(input.getCallChain().get(0));
        }
        if (input.getSinkMethod() == null || input.getSinkMethod().isEmpty()) {
            input.setSinkMethod(input.getCallChain().get(input.getCallChain().size() - 1));
        }

        return input;
    }

    private String asString(Object obj) {
        return obj == null ? null : String.valueOf(obj);
    }

    private Integer asInteger(Object obj) {
        if (obj == null) return null;
        if (obj instanceof Integer) return (Integer) obj;
        return Integer.parseInt(String.valueOf(obj));
    }

    private Boolean asBoolean(Object obj) {
        if (obj == null) return null;
        if (obj instanceof Boolean) return (Boolean) obj;
        return Boolean.parseBoolean(String.valueOf(obj));
    }

    private Double asDouble(Object obj) {
        if (obj == null) return null;
        if (obj instanceof Double) return (Double) obj;
        if (obj instanceof Number) return ((Number) obj).doubleValue();
        return Double.parseDouble(String.valueOf(obj));
    }

    @Data
    private static class UnifiedInput {
        private String cveName;
        private String chainId;
        private List<String> callChain = Collections.emptyList();
        private String entryMethod;
        private String sinkMethod;
        private FinalPayload finalPayload;
        private Integer numDecisions;
        private Integer numInclude;
        private Integer numExclude;
        private Boolean truncated;
        private Double elapsedSec;
        private Map<String, Object> rawInput = Collections.emptyMap();
    }

    @Data
    private static class FinalPayload {
        private Map<String, Object> argValues = Collections.emptyMap();
        private Map<String, Object> fieldValues = Collections.emptyMap();
        private String description;
    }

    @Data
    private static class ResolvedOracle {
        private final MetaInfo metaInfo;
        private final TestcaseUnit testcaseUnit;
    }
}
