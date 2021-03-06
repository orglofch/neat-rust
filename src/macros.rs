#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6,
                "{} is not approximately equal to {}", *a, *b);
    })
}

#[macro_export]
macro_rules! hashmap {
    ( $($key:expr => $value:expr),* ) => ({
        let mut hashmap = ::std::collections::HashMap::new();
        $(
            hashmap.insert($key, $value);
        )+
        hashmap
    })
}
