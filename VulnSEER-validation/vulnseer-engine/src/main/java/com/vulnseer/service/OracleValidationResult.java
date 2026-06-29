package com.vulnseer.service;

import com.vulnseer.testcase.model.TestcaseUnit;

public class OracleValidationResult {

    public enum Status {
        NOT_REACHED,
        INPUT_MISMATCH,
        TIMEOUT_TRIGGERED,
        ORACLE_TRIGGERED,
        ORACLE_NOT_TRIGGERED,
        ORACLE_MISSING,
        ORACLE_ERROR
    }

    private final Status status;
    private final boolean triggered;
    private final Throwable validateThrow;
    private final String message;
    private final TestcaseUnit testcaseUnit;

    private OracleValidationResult(Status status,
                                   boolean triggered,
                                   Throwable validateThrow,
                                   String message,
                                   TestcaseUnit testcaseUnit) {
        this.status = status;
        this.triggered = triggered;
        this.validateThrow = validateThrow;
        this.message = message;
        this.testcaseUnit = testcaseUnit;
    }

    public static OracleValidationResult notReached() {
        return new OracleValidationResult(Status.NOT_REACHED, false, null,
                "Sink method not reached.", null);
    }

    public static OracleValidationResult inputMismatch(TestcaseUnit testcaseUnit) {
        return new OracleValidationResult(Status.INPUT_MISMATCH, false, null,
                "Reached sink input does not match the groundtruth validInput.", testcaseUnit);
    }

    public static OracleValidationResult timeoutTriggered(Throwable validateThrow, TestcaseUnit testcaseUnit) {
        return new OracleValidationResult(Status.TIMEOUT_TRIGGERED, true, validateThrow,
                "Timeout is treated as a vulnerability trigger by the legacy oracle.", testcaseUnit);
    }

    public static OracleValidationResult triggered(Throwable validateThrow, TestcaseUnit testcaseUnit) {
        return new OracleValidationResult(Status.ORACLE_TRIGGERED, true, validateThrow,
                "Groundtruth validator assertion failed.", testcaseUnit);
    }

    public static OracleValidationResult notTriggered(TestcaseUnit testcaseUnit) {
        return new OracleValidationResult(Status.ORACLE_NOT_TRIGGERED, false, null,
                "Groundtruth validator completed without assertion failure.", testcaseUnit);
    }

    public static OracleValidationResult missing(String message) {
        return new OracleValidationResult(Status.ORACLE_MISSING, false, null, message, null);
    }

    public static OracleValidationResult error(TestcaseUnit testcaseUnit, Throwable throwable) {
        return new OracleValidationResult(Status.ORACLE_ERROR, false, throwable,
                throwable == null ? "Oracle validation error." : throwable.toString(), testcaseUnit);
    }

    public Status getStatus() {
        return status;
    }

    public boolean isTriggered() {
        return triggered;
    }

    public Throwable getValidateThrow() {
        return validateThrow;
    }

    public String getMessage() {
        return message;
    }

    public TestcaseUnit getTestcaseUnit() {
        return testcaseUnit;
    }
}
