package com.vulnseer.service;

import com.vulnseer.fuzz.result.InvokeMethodResult;
import com.vulnseer.fuzz.result.ResultStatus;
import com.vulnseer.testcase.model.TestcaseUnit;
import com.vulnseer.util.ReflectionUtil;
import org.junit.runners.model.TestTimedOutException;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URI;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

public class OracleValidationService {

    public static final long DEFAULT_TIMEOUT = 90000L;
    public static final TimeUnit DEFAULT_TIME_UNIT = TimeUnit.MILLISECONDS;

    public OracleValidationResult validate(ClassLoader classLoader,
                                           TestcaseUnit testcaseUnit,
                                           InvokeMethodResult result,
                                           Object[] expectedValidInput,
                                           boolean enforceInputCheck) {
        return validate(classLoader, testcaseUnit, result, expectedValidInput,
                enforceInputCheck, DEFAULT_TIMEOUT, DEFAULT_TIME_UNIT);
    }

    public OracleValidationResult validate(ClassLoader classLoader,
                                           TestcaseUnit testcaseUnit,
                                           InvokeMethodResult result,
                                           Object[] expectedValidInput,
                                           boolean enforceInputCheck,
                                           long timeout,
                                           TimeUnit timeUnit) {
        if (testcaseUnit == null) {
            return OracleValidationResult.missing("No matching groundtruth testcase was found.");
        }
        if (result == null || ResultStatus.NotReached.equals(result.getResultStatus())) {
            return OracleValidationResult.notReached();
        }
        if (enforceInputCheck && !checkInputValid(expectedValidInput, result.getInputParams())) {
            return OracleValidationResult.inputMismatch(testcaseUnit);
        }
        if (result.getThrowValue() instanceof TestTimedOutException) {
            return OracleValidationResult.timeoutTriggered(
                    new TestTimedOutException(timeout, timeUnit), testcaseUnit);
        }
        if (!testcaseUnit.isNeedValidateReturnValue() && !testcaseUnit.isNeedValidateThrow()) {
            return OracleValidationResult.missing("The matching testcase does not configure a return/throw oracle.");
        }

        try {
            Class<?> clazz = classLoader.loadClass(testcaseUnit.getTestcaseClassName());
            Object validator = clazz.getDeclaredConstructor().newInstance();

            boolean returnValueValidateStatus = false;
            boolean throwValidateStatus = false;
            Throwable validateThrow = null;

            if (testcaseUnit.isNeedValidateReturnValue()) {
                OracleMethodResult methodResult = invokeOracleMethod(
                        clazz, validator, testcaseUnit.getValidateReturnValueMethodName(), result.getReturnValue());
                if (methodResult.isError()) {
                    return OracleValidationResult.error(testcaseUnit, methodResult.getThrowable());
                }
                returnValueValidateStatus = methodResult.isAssertionFailed();
                if (returnValueValidateStatus) {
                    validateThrow = methodResult.getThrowable();
                }
            }

            if (testcaseUnit.isNeedValidateThrow()) {
                OracleMethodResult methodResult = invokeOracleMethod(
                        clazz, validator, testcaseUnit.getValidateThrowMethodName(), result.getThrowValue());
                if (methodResult.isError()) {
                    return OracleValidationResult.error(testcaseUnit, methodResult.getThrowable());
                }
                throwValidateStatus = methodResult.isAssertionFailed();
                if (throwValidateStatus) {
                    validateThrow = methodResult.getThrowable();
                }
            }

            boolean triggered;
            if (testcaseUnit.isNeedValidateThrow() && testcaseUnit.isNeedValidateReturnValue()) {
                triggered = returnValueValidateStatus && throwValidateStatus;
            } else if (testcaseUnit.isNeedValidateReturnValue()) {
                triggered = returnValueValidateStatus;
            } else {
                triggered = throwValidateStatus;
            }

            if (triggered) {
                return OracleValidationResult.triggered(validateThrow, testcaseUnit);
            }
            return OracleValidationResult.notTriggered(testcaseUnit);
        } catch (Throwable throwable) {
            return OracleValidationResult.error(testcaseUnit, throwable);
        }
    }

    private OracleMethodResult invokeOracleMethod(Class<?> clazz,
                                                  Object validator,
                                                  String methodName,
                                                  Object input) {
        Method method = ReflectionUtil.getMethodByName(clazz, methodName);
        if (method == null) {
            return OracleMethodResult.error(new NoSuchMethodException(methodName));
        }

        ReflectionUtil.setAccessible(method);
        try {
            method.invoke(validator, input);
            return OracleMethodResult.notTriggered();
        } catch (InvocationTargetException e) {
            Throwable targetException = e.getTargetException();
            if (targetException instanceof AssertionError) {
                return OracleMethodResult.triggered(targetException);
            }
            return OracleMethodResult.error(targetException == null ? e : targetException);
        } catch (Throwable throwable) {
            return OracleMethodResult.error(throwable);
        }
    }

    private boolean checkInputValid(Object[] validInputs, Object[] inputParams) {
        if (validInputs == null || inputParams == null) {
            return false;
        }
        if (validInputs.length != inputParams.length) {
            return false;
        }
        for (int i = 0; i < validInputs.length; i++) {
            if (validInputs[i] == null) {
                if (inputParams[i] != null) {
                    return false;
                }
            } else {
                if (validInputs[i] instanceof String || validInputs[i] instanceof URI) {
                    if (!Objects.equals(validInputs[i], inputParams[i])) {
                        return false;
                    }
                }
                if (validInputs[i].getClass().isPrimitive()) {
                    if (validInputs[i] != inputParams[i]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private static class OracleMethodResult {
        private final boolean assertionFailed;
        private final boolean error;
        private final Throwable throwable;

        private OracleMethodResult(boolean assertionFailed, boolean error, Throwable throwable) {
            this.assertionFailed = assertionFailed;
            this.error = error;
            this.throwable = throwable;
        }

        private static OracleMethodResult triggered(Throwable throwable) {
            return new OracleMethodResult(true, false, throwable);
        }

        private static OracleMethodResult notTriggered() {
            return new OracleMethodResult(false, false, null);
        }

        private static OracleMethodResult error(Throwable throwable) {
            return new OracleMethodResult(false, true, throwable);
        }

        private boolean isAssertionFailed() {
            return assertionFailed;
        }

        private boolean isError() {
            return error;
        }

        private Throwable getThrowable() {
            return throwable;
        }
    }
}
